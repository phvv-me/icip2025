import os
import re
from functools import cached_property

import humanize
import torch
import transformers
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AwqConfig,
)
from transformers.utils import is_auto_awq_available

from ..abstract import BaseModel
from ..utils.stdlib import is_online
from ..utils.tensor import unsqueeze_like

Q4BIT = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

Q8BIT = transformers.BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
)


AWQ = (
    AwqConfig(
        bits=4,
        fuse_max_seq_len=512,  # Note: Update this as per your use-case
        do_fuse=True,
    )
    if is_auto_awq_available() and torch.cuda.is_available()
    else None
)


class HuggingFaceModel(BaseModel, arbitrary_types_allowed=True):
    id: str
    family: str | None = None

    cls: type[AutoModelForCausalLM] = AutoModelForCausalLM
    tkn: type[AutoTokenizer] = AutoTokenizer

    device_map: int | str = "auto"
    torch_dtype: torch.dtype | str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str = "auto"
    quantization: int | str | None = "auto"
    compile: bool = False

    _quantizations = {4: Q4BIT, 8: Q8BIT, "AWQ": AWQ}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load()
        print("Loaded model", self.id)

    def __call__(self, *args, **kwds):
        return self._model(*args, **kwds)

    def __str__(self):
        return self.id if self.family is None else self.name

    def load(self) -> None:
        self.login_to_hf_if_online()

        self._model = AutoModelForCausalLM.from_pretrained(**self._model_kwargs())
        self._tokenizer = AutoTokenizer.from_pretrained(**self._tokenizer_kwargs())

        if self.compile:
            self._compile()

        if "OpenELM" in self.id:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self._tokenizer = tokenizer

        if "Meta-Llama-3" in self.id:
            self._fix_pad_token_in_llama_model()
            self._fix_eos_token_in_llama_model()
            self._tokenizer.padding_side = "left"

        if self.id.startswith("mistalai/Mistral"):
            self._fix_pad_token_in_mistral_model()

    def _model_kwargs(self):
        kwargs = dict(
            pretrained_model_name_or_path=self.id,
            device_map=self.device_map,
            torch_dtype=self._correct_torch_dtype,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self._get_attn_implementation(),
            local_files_only=not is_online(),
        )

        if quantization_config := self._get_quantization_config():
            kwargs["quantization_config"] = quantization_config

        return kwargs

    def _tokenizer_kwargs(self):
        return dict(
            pretrained_model_name_or_path=self.id,
            trust_remote_code=self.trust_remote_code,
            local_files_only=not is_online(),
        )

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model(self) -> AutoModelForCausalLM:
        return self._model

    @property
    def memory_footprint(self) -> int:
        """Returns the current CUDA memory cost of the model in bytes."""
        return self._model.get_memory_footprint()

    @cached_property
    def parameter_count(self) -> int:
        """Returns the number of parameters in the model."""
        return sum(p.numel() for p in self._model.parameters())

    @property
    def is_instruction_finetuned(self) -> bool:
        return self.id.lower().endswith("it") or "instruct" in self.id.lower()

    @property
    def _parameterc_count_string_extracted_from_id(self):
        return re.compile(r"(\d+[Bb])").search(self.id).group(1).upper()

    @property
    def _parameter_count_string_from_actual_count(self):
        word = humanize.intword(self.parameter_count, format="%.f")
        i_alpha = next(i for i, char in enumerate(word) if char.isalpha())
        return word[: i_alpha + 1].upper().replace(" ", "")

    @property
    def parameter_count_string(self) -> str:
        try:
            return self._parameterc_count_string_extracted_from_id
        except AttributeError:
            return self._parameter_count_string_from_actual_count

    @property
    def is_instruction_finetuned_string(self):
        return "(Instruct)" if self.is_instruction_finetuned else ""

    @property
    def property_string(self):
        return f"{self.parameter_count_string} {self.is_instruction_finetuned_string}".strip()

    @property
    def name(self):
        return f"{self.family} {self.property_string}"

    @property
    def device(self) -> torch.device:
        """Returns the device the model is allocated to."""
        return self._model.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the model data type."""
        return self._model.dtype

    @property
    def get_embeddings(self) -> torch.Tensor:
        return self._model.get_input_embeddings()

    @property
    def unembedding_matrix(self) -> torch.Tensor:
        return self._model.lm_head.weight.data.detach()

    @property
    def _correct_torch_dtype(self) -> torch.dtype:
        if self.torch_dtype == "auto" and "AWQ" in self.id.upper():
            return torch.float16
        return self.torch_dtype

    def _get_quantization_config(self):
        return self._quantizations.get(self.quantization)

    def _get_attn_implementation(self):
        if self.attn_implementation != "auto":
            return self.attn_implementation
        return "flash_attention_2" if torch.cuda.is_available() else "eager"

    def _fix_token(self, token_str, token_type):
        token_id = self._tokenizer.convert_tokens_to_ids(token_str)
        setattr(self._tokenizer, f"{token_type}_token", token_str)
        setattr(self._tokenizer, f"{token_type}_token_id", token_id)

    def _fix_pad_token_in_llama_model(self):
        self._fix_token("<|eot_id|>", "pad")

    def _fix_eos_token_in_llama_model(self):
        self._fix_token("<|end_of_text|>", "eos")

    def _fix_pad_token_in_mistral_model(self):
        self._fix_token(self._tokenizer.eos_token, "pad")

    def make_input(
        self, inputs: str | torch.IntTensor, *args, **kwargs
    ) -> torch.Tensor:
        return self._tokenizer(
            text=(
                self.decode(inputs.flatten(0, 1))
                if isinstance(inputs, torch.Tensor)
                else inputs
            ),
            return_tensors="pt",
            *args,
            **kwargs,
        ).to(self.device)

    def tokenize(self, *args, **kwargs) -> torch.Tensor:
        return self.make_input(*args, **kwargs)["input_ids"]

    def decode(self, input_ids: torch.Tensor) -> str:
        decoded = self._tokenizer.batch_decode(input_ids)
        return [self._clean(x) for x in decoded]

    def _clean(self, text: str) -> str:
        """Clean the input text."""
        return (
            text.replace(self._tokenizer.bos_token, "")
            .replace(self._tokenizer.pad_token, "")
            .lstrip()
        )

    def embed(self, input_text: str) -> torch.Tensor:
        return self.get_embeddings(self.tokenize(input_text))

    def forward_last_hiden_state(self, *args, **kwargs) -> torch.Tensor:
        return self._model(*args, output_hidden_states=True, **kwargs).hidden_states[-1]

    def forward_next_token_logit(
        self, inputs_embeds: torch.FloatTensor, probes: torch.FloatTensor
    ) -> torch.Tensor:
        hs = self.forward_last_hiden_state(inputs_embeds=inputs_embeds)
        return hs.matmul(probes)[..., -1, :]

    def linear_path(self, input_text: str, steps: int = 8):
        x = self.embed(input_text)
        x0 = self.embed("".join([self._tokenizer.pad_token] * (x.size(1) - 1)))
        alphas = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)
        return x0 + unsqueeze_like(alphas, x) * (x - x0)

    def integrated_gradients(
        self,
        input_text: str,
        probes: torch.FloatTensor,
        steps: int = 8,
        *args,
        **kwargs,
    ) -> torch.return_types.max:
        x = self.embed(input_text)
        x0 = self.embed("".join([self._tokenizer.pad_token] * (x.size(1) - 1)))
        alphas = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)
        x_path = x0 + unsqueeze_like(alphas, x) * (x - x0)

        def forward_func(x):
            return self.forward_next_token_logit(x, probes)

        grads = [
            torch.autograd.functional.jacobian(
                forward_func,
                inputs=x_i.unsqueeze(0).requires_grad_(True),
                *args,
                **kwargs,
            )
            .diagonal(dim1=0, dim2=-3)
            .squeeze(0)
            for x_i in tqdm(x_path, desc="Computing Jacobian")
        ]

        # Stack the gradients
        grads = torch.stack(grads)

        # Compute the trapezoid rule
        ig = torch.trapz(grads, dx=1 / steps, dim=0).sum(-1)

        return ig

    def _compile(self):
        self._model.forward = torch.compile(
            self._model.forward, mode="reduce-overhead", fullgraph=True
        )

    @staticmethod
    def is_hf_online():
        return not bool(os.getenv("HF_HUB_OFFLINE"))

    @classmethod
    def login_to_hf_if_online(cls):
        if is_online() and cls.is_hf_online():
            login(os.environ["HUGGING_FACE_LOGIN_TOKEN"], add_to_git_credential=True)

    # vectorizable version, but slower otherwise
    # def integrated_gradients(
    #     self, input_text: str, probes: torch.FloatTensor, steps: int = 8, *args, **kwargs
    # ) -> torch.return_types.max:
    #     x = self.embed(input_text)
    #     x0 = self.embed("".join([self.tokenizer.pad_token] * (x.size(1) - 1)))
    #     alphas = torch.linspace(0, 1, steps, device=x.device, dtype=x.dtype)

    #     x_path = x0 + unsqueeze_like(alphas, x) * (x - x0)

    #     grads = torch.autograd.functional.jacobian(
    #         lambda x: self.forward_next_token_logit(x, probes), inputs=x_path, *args, **kwargs
    #     ).diagonal(dim1=0, dim2=-3)

    #     return torch.trapz(grads, dx=1 / steps, dim=-1).sum(-1)
