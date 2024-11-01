from typing import Type

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    PreTrainedModel,
)

from ...utils.stdlib import is_online
from .base import BaseHuggingFaceModel


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
        )

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def unembedding_matrix(self) -> torch.Tensor:
        return self._model.language_model.lm_head.weight.data.detach()

    def generate(self, image: Image.Image, text: str, *args, **kwargs):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": text}],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            image, input_text, add_special_tokens=True, return_tensors="pt"
        ).to(self.model.device)

        return self.model.generate(*args, **inputs, **kwargs)

    def decode(self, output: torch.Tensor):
        return self.processor.batch_decode(output, skip_special_tokens=False)
