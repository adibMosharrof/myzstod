from datamodules.data_collators.cross_collator import CrossCollator
from my_enums import ContextType
from typing import Union
from datamodules.data_collators.t5_collator import T5Collator
from datamodules.data_collators.decoder_collator import DecoderCollator
from datamodules.data_collators.base_collator import BaseCollator
from transformers import AutoTokenizer
import utils


class CollatorFactory:
    @staticmethod
    def create_collator(
        model_name: str,
        context_type: ContextType,
        tokenizer: AutoTokenizer,
        prompt_cls,
        max_token_len: int = 1024,
        test_prompt_max_len: int = 750,
        schema_max_len: int = 350,
    ) -> BaseCollator:
        if context_type == ContextType.GPT_CROSS:
            return CrossCollator(
                tokenizer=tokenizer,
                nlg_prompt_cls=prompt_cls,
                max_token_len=max_token_len,
                test_prompt_max_len=test_prompt_max_len,
                schema_max_length=schema_max_len,
            )
        if utils.is_t5_model(model_name):
            return T5Collator(
                tokenizer=tokenizer,
                nlg_prompt_cls=prompt_cls,
                max_token_len=max_token_len,
                test_prompt_max_len=test_prompt_max_len,
            )
        return DecoderCollator(
            tokenizer=tokenizer,
            nlg_prompt_cls=prompt_cls,
            max_token_len=max_token_len,
            test_prompt_max_len=test_prompt_max_len,
        )
