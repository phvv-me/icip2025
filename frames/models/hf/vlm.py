
from typing import Any, Type, TypedDict

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    PreTrainedModel,
    Qwen2VLForConditionalGeneration,
    LlavaForConditionalGeneration
)

from ...utils.stdlib import is_online
from .base import BaseHuggingFaceModel

class VLMInput(TypedDict):
    text: str
    image: Image.Image


class VisionLanguageHuggingFaceModel(BaseHuggingFaceModel):
    cls: Type[PreTrainedModel] | None = None
    prc: Type[AutoProcessor] = AutoProcessor

    def load(self) -> None:
        self._model = self._auto_cls.from_pretrained(**self._model_kwargs())
        self._processor = self.prc.from_pretrained(**self._processor_kwargs())
        self._tokenizer = self.processor.tokenizer

    def _processor_kwargs(self):
        return dict(
            pretrained_model_name_or_path=self.id,
            trust_remote_code=self.trust_remote_code,
            local_files_only=not is_online(),
            padding_side='left',
        )

    @property
    def _auto_cls(self) -> type[PreTrainedModel]:
        if self._is_meta_llama():
            return MllamaForConditionalGeneration
        elif self._is_qwen2vl():
            return Qwen2VLForConditionalGeneration
        elif self._is_pixtral() or self._is_llava():
            return LlavaForConditionalGeneration
        else:
            raise ValueError(f"Model {self.id} not supported")

    def _is_qwen2vl(self) -> bool:
        return "qwen2-vl" in self.id.lower()

    def _is_pixtral(self) -> bool:
        return "pixtral" in self.id.lower()

    def _is_llava(self) -> bool:
        return "llava" in self.id.lower()

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def unembedding_matrix(self) -> torch.Tensor:
        if self._is_meta_llama():
            return self.model.language_model.lm_head.weight.data.detach()
        elif self._is_qwen2vl():
            return self.model.lm_head.weight.data.detach()
        else:
            raise ValueError(f"Model {self.id} not supported")

    def generate(self, text: str, image: Image.Image | None = None, *args, **kwargs):
        inputs = self.make_input(text, image)
        return self.model.generate(*args, **inputs, **kwargs)

    def make_input(self, text: str, image: Image.Image | None = None, *args, **kwargs):
        text = self._make_text_input(text, has_image=image is not None)
        return self.processor(image, text, add_special_tokens=True, return_tensors="pt", *args, **kwargs).to(self.model.device)

    def _make_text_input(self, text: str, has_image: bool = False):
        messages = self._build_simple_message(text, has_image)
        input_text = self._convert_to_chat_template(messages)
        return input_text

    def _convert_to_chat_template(self, messages):
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _build_simple_message(self, text, has_image: bool = False):
        content = [{"type": "image"}, {"type": "text", "text": text}] if has_image else [{"type": "text", "text": text}]
        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def decode(self, output: torch.Tensor):
        return self.processor.batch_decode(output, skip_special_tokens=False)
