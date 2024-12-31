from functools import cache, cached_property
import json
from typing import Optional
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
from ..hf import LanguageHuggingFaceModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

class HuggingFaceLLMDeepEval(DeepEvalBaseLLM):
    def __init__(self, **kwargs):
        self.hf_model = LanguageHuggingFaceModel(**kwargs)

    @cache
    def load_model(self):
        self.hf_model.load()

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        output_dict = self.pipeline(prompt, prefix_allowed_tokens_fn=self._parse_schema(schema))
        output = output_dict[0]["generated_text"][len(prompt):]
        json_result = json.loads(output)
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.hf_model.id

    def _parse_schema(self, schema: BaseModel):
        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.model_json_schema())
        return build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)

    @cached_property
    def pipeline(self):
        return pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="cuda:0",
            # padding="max_length",
            max_new_tokens=1024,
            do_sample=False,
            top_k=None,
            top_p=None,
            temperature=None,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    @property
    def model(self):
        return self.hf_model.model

    @property
    def tokenizer(self):
        return self.hf_model.tokenizer