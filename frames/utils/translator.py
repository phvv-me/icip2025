import asyncio
from itertools import batched
from typing import Iterable, List, Union

import pandas as pd
import pydantic
from googletrans import LANGUAGES, Translator
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


class MultiTranslator(pydantic.BaseModel):
    batch_size: int = 10
    separator: str = "\t\n\n"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._translator: Translator = Translator()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_cls=Exception,
    )
    async def _translate_text(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        translation = await self._translator.translate(text, target_lang, source_lang)
        return translation.text

    async def _translate_batch_together(
        self, batch: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        batch_text = self.separator.join(batch)
        translation = await self._translate_text(batch_text, source_lang, target_lang)
        translations = translation.split(self.separator)

        if len(translations) != len(batch):
            raise ValueError(
                "Batch translation resulted in incorrect number of translations"
            )

        return translations

    async def _process_batch(
        self, batch: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        try:
            return await self._translate_batch_together(batch, source_lang, target_lang)
        except Exception as e:
            logger.warning(
                f"Batch translation failed, falling back to individual: {str(e)}"
            )
            return await asyncio.gather(
                *[
                    self._translate_text(text, source_lang, target_lang)
                    for text in batch
                ]
            )

    async def _translate_to_language(
        self, sentences: List[str], target_lang: str, source_lang: str
    ) -> List[str]:
        logger.info(f"Translating {len(sentences)} sentences to {target_lang}")
        all_batches = batched(tqdm(sentences, desc=f"→ {target_lang}"), self.batch_size)
        translations = []

        for batch in all_batches:
            batch_translations = await self._process_batch(
                list(batch), source_lang, target_lang
            )
            translations.extend(batch_translations)

        return translations

    async def translate(
        self,
        sentences: Iterable[str],
        target_langs: Union[str, List[str]],
        source_lang: str = "auto",
    ) -> pd.DataFrame:
        sentences_list = list(sentences)
        target_langs_list = (
            [target_langs] if isinstance(target_langs, str) else target_langs
        )

        translations = {}
        for lang in target_langs_list:
            lang_name = LANGUAGES[lang]
            translations[lang_name] = await self._translate_to_language(
                sentences_list, lang, source_lang
            )

        return pd.DataFrame({"original": sentences_list} | translations)
