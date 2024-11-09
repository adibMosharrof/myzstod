from transformers import AutoTokenizer
import torch

from my_enums import SpecialTokens
from utilities.context_manager import ContextManager


class TokenizerUtilities:
    @staticmethod
    def tokenize(tokenizer: AutoTokenizer, text: str, max_len: int = None):
        tokens = tokenizer.encode(text, return_tensors="pt", max_length=max_len)
        return tokens.to(dtype=torch.int32)[0]

    @staticmethod
    def tokenize_with_pad(tokenizer: AutoTokenizer, text: str, max_len: int = None):
        return tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )

    @staticmethod
    def get_tokenizer(
        tokenizer_name: str = None, model_name: str = None, context_type: str = None
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name or model_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        special_tokens = ["{", "}"]
        if any(
            [
                ContextManager.is_zstod(context_type),
                ContextManager.is_simple_tod(context_type),
            ]
        ):
            special_tokens += SpecialTokens.list()
        else:
            special_tokens += [SpecialTokens.user.value, SpecialTokens.system.value]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        return tokenizer
