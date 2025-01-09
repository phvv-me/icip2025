import sys
import os
from googletrans import Translator, LANGUAGES
import logging
import warnings
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import time

@dataclass
class TranslationConfig:
    """Configuration for translation process"""
    batch_size: int = 10
    delay_between_batches: float = 1.5
    retry_delay: float = 3.0
    single_translation_delay: float = 0.5
    separator: str = " ; "
    max_retries: int = 3

class MultiTranslator:
    def __init__(self, config: Optional[TranslationConfig] = None):
        self.config = config or TranslationConfig()
        self.translator = Translator()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_output_path(self, input_file: str, target_lang: str) -> str:
        input_path = Path(input_file)
        return str(input_path.parent / f"{input_path.stem}_{target_lang}{input_path.suffix}")
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single piece of text"""
        try:
            translation = self.translator.translate(text, src=source_lang, dest=target_lang)
            return translation.text
        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return f"ERROR: {text}"
    
    def translate_sentences(
        self,
        input_file: str,
        source_lang: str = 'auto',
        target_lang: str = 'en'
    ) -> str:
        output_file = self.get_output_path(input_file, target_lang)
        translated_lines = []
        current_batch = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            all_lines = [line.strip() for line in f if line.strip()]
            
        total_lines = len(all_lines)
        self.logger.info(f"Translating {total_lines} sentences to {target_lang} ({LANGUAGES.get(target_lang, 'Unknown')})")
        
        try:
            for i, line in enumerate(all_lines):
                current_batch.append(line)
                
                if len(current_batch) == self.config.batch_size or i == total_lines - 1:
                    try:
                        # Translate batch
                        batch_text = self.config.separator.join(current_batch)
                        translation_text = self.translate_text(
                            batch_text,
                            source_lang,
                            target_lang
                        )
                        
                        translated_batch = translation_text.split(self.config.separator)
                        
                        # Handle batch size mismatch
                        if len(translated_batch) != len(current_batch):
                            translated_batch = []
                            for single_line in current_batch:
                                single_translation = self.translate_text(
                                    single_line,
                                    source_lang,
                                    target_lang
                                )
                                translated_batch.append(single_translation)
                                time.sleep(self.config.single_translation_delay)
                                
                        translated_lines.extend(translated_batch)
                        self.logger.info(f"Progress: {min(i + 1, total_lines)}/{total_lines} sentences")
                        
                        current_batch = []
                        time.sleep(self.config.delay_between_batches)
                        
                    except Exception as e:
                        self.logger.error(f"Batch translation error: {str(e)}")
                        for original in current_batch:
                            translated_lines.append(f"ERROR: {original}")
                        current_batch = []
                        time.sleep(self.config.retry_delay)
                        
        except KeyboardInterrupt:
            self.logger.info("Translation interrupted by user")
            
        finally:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in translated_lines:
                    f.write(line + '\n')
                    
            if len(translated_lines) > 0:
                self.logger.info(f"Translation completed. Saved to: {output_file}")
                
        return output_file

def translate_to_languages(
    input_file: str,
    target_languages: List[str],
    source_lang: str = 'auto',
    config: Optional[TranslationConfig] = None
) -> Dict[str, str]:
    """
    Translate file to multiple languages
    
    Args:
        input_file: Path to input file
        target_languages: List of language codes to translate to
        source_lang: Source language code (default: 'auto')
        config: Optional translation configuration
        
    Returns:
        Dict[str, str]: Dictionary mapping language codes to output file paths
    """
    translator = MultiTranslator(config)
    results = {}
    
    for lang in target_languages:
        try:
            output_file = translator.translate_sentences(
                input_file=input_file,
                source_lang=source_lang,
                target_lang=lang
            )
            results[lang] = output_file
        except Exception as e:
            logging.error(f"Error translating to {lang}: {str(e)}")
    
    return results

# Example usage in Jupyter:
"""
# First, install required packages if you haven't:
# !pip install googletrans==3.1.0a0

# Custom configuration
config = TranslationConfig(
    batch_size=10,
    delay_between_batches=1.5,
    retry_delay=3.0,
    single_translation_delay=0.5,
    separator=" ; "
)

# List of target languages
target_langs = ['id']  # Indonesian

# Translate to multiple languages
results = translate_to_languages(
    input_file="path/to/your/input/file.txt",
    target_languages=target_langs,
    source_lang='en',
    config=config
)

# Print results
for lang, output_file in results.items():
    print(f"Translations to {lang} ({LANGUAGES.get(lang)}): {output_file}")
"""