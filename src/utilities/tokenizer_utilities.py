from transformers import AutoTokenizer
import torch


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
