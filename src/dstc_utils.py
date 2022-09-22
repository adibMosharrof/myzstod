import glob
import re
from pathlib import Path
from typing import List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tokenizers.processors import TemplateProcessing

from my_enums import SimpleTodConstants, SpecialTokens


def get_dstc_service_name(service_name: str) -> str:
    return service_name[: service_name.find("_")]


def get_dialog_file_paths(data_root, step):
    pattern = "dialogues_*"
    file_paths = glob.glob(str(data_root / step / pattern))
    return file_paths


def get_csv_data_path(
    step: str = "train",
    num_dialogs: int = 1,
    delexicalized: bool = True,
    processed_data_root: Path = None,
    domains: List[str] = None,
    num_turns: int = 26,
    is_multi_task: bool = False,
    should_add_schema: bool = False,
):
    step_dir = processed_data_root / step
    return (
        step_dir
        / f"simple_tod_dstc_multi_task_{is_multi_task}_schema_{should_add_schema}_turns_{num_turns}_dialogs_{num_dialogs}{SimpleTodConstants.DELEXICALIZED if delexicalized else ''}_{'_'.join(domains)}.csv"
    )


def get_tokenizer(model_name: str = "gpt2") -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side="left",
        pad_token=SpecialTokens.pad_token.value,
        bos_token=SpecialTokens.bos_token.value,
        eos_token=SpecialTokens.eos_token.value,
        additional_special_tokens=SpecialTokens.list(),
    )
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{tokenizer.bos_token}:0 $A:0 {tokenizer.eos_token}:0",
        special_tokens=[
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        ],
    )
    return tokenizer


def get_token_id(tokenizer: AutoTokenizer, token_str: str) -> int:
    return tokenizer(token_str)["input_ids"][0]


def get_text_in_between(
    text: str, start_token: str, end_token: str, default_value: any = None
) -> Optional[str]:
    try:
        idx1 = text.index(start_token)
        idx2 = text.index(end_token)
        res = text[idx1 + len(start_token) : idx2]
        return res
    except ValueError:
        return default_value


def remove_tokens_from_text(text: str, tokens: List[str]) -> str:
    for token in tokens:
        text = text.replace(token, "")
    return text
