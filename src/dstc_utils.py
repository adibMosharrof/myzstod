import glob
import re
from pathlib import Path
from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from simple_tod_dataclasses import SimpleTodConstants, SpecialTokens, TokenizerTokens
from tokenizers.processors import TemplateProcessing


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
):
    step_dir = processed_data_root / step
    return (
        step_dir
        / f"simple_tod_dstc_multi_task_{is_multi_task}_turns_{num_turns}_dialogs_{num_dialogs}{SimpleTodConstants.DELEXICALIZED if delexicalized else ''}_{'_'.join(domains)}.csv"
    )


def get_tokenizer(
    model_name: str = "gpt2", max_token_len: int = 700
) -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_token_len,
        truncation_side="left",
        pad_token=TokenizerTokens.pad_token,
        bos_token=TokenizerTokens.bos_token,
        eos_token=TokenizerTokens.eos_token,
        # bos_token=SpecialTokens.begin_context,
        # eos_token=SpecialTokens.end_response,
    )
    special_tokens = SpecialTokens.list()
    tokenizer.add_tokens(special_tokens, special_tokens=True)
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
