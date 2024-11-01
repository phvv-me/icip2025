"""
Module for working with HuggingFace models, providing a wrapper class with quantization support.
"""

import os
from typing import Type

import torch
from huggingface_hub import login
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

from ...utils.stdlib import is_online
from ...utils.tensor import unsqueeze_like
from .base import BaseHuggingFaceModel


class LanguageHuggingFaceModel(BaseHuggingFaceModel):
    tkn: Type[PreTrainedTokenizer] = AutoTokenizer

    def load(self) -> None:
        """
        Load the model and tokenizer from HuggingFace Hub.
        Handles login if online and sets up the model with specified configuration.
        """
        super().load()
        self._tokenizer = self.tkn.from_pretrained(**self._tokenizer_kwargs())

        if "OpenELM" in self.id:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            self._tokenizer = tokenizer

        if self.id.startswith("mistalai/Mistral"):
            self._fix_pad_token_in_mistral_model()

    def _tokenizer_kwargs(self):
        return dict(
            pretrained_model_name_or_path=self.id,
            trust_remote_code=self.trust_remote_code,
            local_files_only=not is_online(),
        )

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
