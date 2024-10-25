import torch
from datamodules.data_collators.base_collator import BaseCollator
from my_enums import ZsTodConstants
import torch.nn.functional as F


class ApiInContextFilter:
    def __init__(self, collator: BaseCollator) -> None:
        self.collator = collator

    def apply(self, data: list) -> list:
        out = []
        for item in data:
            context_text = self.collator.get_context_text(item)
            context_tokens, context_unused_len = (
                self.collator.get_context_tokens_and_unused_len(item, context_text)
            )
            is_api_in_context = self.is_api_call_present(context_tokens)
            if is_api_in_context:
                out.append(item)
        return out

    def is_api_call_present(self, context_tokens):
        context_text = self.collator.tokenizer.decode(context_tokens)
        return ZsTodConstants.API_CALL.value in context_text
