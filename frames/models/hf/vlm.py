from typing import Any, Type, TypedDict

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    PreTrainedModel,
)

from ...utils.stdlib import is_online
from .base import BaseHuggingFaceModel

class VLMInput(TypedDict):
    text: str
    image: Image.Image


class VisionLanguageHuggingFaceModel(BaseHuggingFaceModel):
    cls: Type[PreTrainedModel] = MllamaForConditionalGeneration
    prc: Type[AutoProcessor] = AutoProcessor

    def load(self) -> None:
        super().load()
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
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def unembedding_matrix(self) -> torch.Tensor:
        return self._model.language_model.lm_head.weight.data.detach()

    def generate(self, text: str, image: Image.Image, *args, **kwargs):
        inputs = self.make_input(text, image)
        return self.model.generate(*args, **inputs, **kwargs)

    def make_input(self, text: str, image: Image.Image, *args, **kwargs):
        text = self._make_text_input(text)
        return self.processor(image, text, add_special_tokens=True, return_tensors="pt", *args, **kwargs).to(self.model.device)

    def _make_text_input(self, text: str):
        messages = self._build_simple_message(text)
        input_text = self._convert_to_chat_template(messages)
        return input_text

    def _convert_to_chat_template(self, messages):
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def _build_simple_message(self, text):
        return [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]

    def decode(self, output: torch.Tensor):
        return self.processor.batch_decode(output, skip_special_tokens=False)
