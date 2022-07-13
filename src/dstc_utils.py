import glob
from pathlib import Path
import re

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from simple_tod_dataclasses import SimpleTodConstants, SpecialTokens


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
    data_root: Path = None,
):
    step_dir = data_root / step
    return (
        step_dir
        / f"simple_tod_dstc_{num_dialogs}{SimpleTodConstants.DELEXICALIZED if delexicalized else ''}.csv"
    )


def get_tokenizer(model_name: str = "gpt2") -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    special_tokens = SpecialTokens.list()
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    return tokenizer
