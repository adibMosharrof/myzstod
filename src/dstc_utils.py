import glob
import re
from pathlib import Path
from typing import List, Optional, Union, Dict
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os
from tokenizers.processors import TemplateProcessing
from hydra_configs import DataModuleConfig, DataPrepConfig

from my_enums import SimpleTodConstants, SpecialTokens
import utils


def get_dstc_service_name(service_name: str) -> str:
    return service_name[: service_name.find("_")]


def get_dialog_file_paths(data_root, step):
    pattern = "dialogues"
    # file_paths = glob.glob(str(data_root / step / pattern))
    files = sorted(os.listdir(data_root / step))
    file_paths = [data_root / step / f for f in files if pattern in f]
    return file_paths


def get_csv_data_path(
    step: str = "train",
    num_dialogs: int = 1,
    cfg: Union[DataPrepConfig, DataModuleConfig] = None,
):
    step_dir = cfg.processed_data_root / step
    return step_dir / (
        "_".join(
            [
                "context_type",
                cfg.context_type,
                "multi_task",
                str(cfg.is_multi_task),
                "_".join(map(str, cfg.multi_tasks)),
                "schema",
                str(cfg.should_add_schema),
                "user_actions",
                str(cfg.should_add_user_actions),
                "sys_actions",
                str(cfg.should_add_sys_actions),
                "turns",
                str(cfg.num_turns),
                "service_results",
                str(cfg.should_add_service_results),
                "dialogs",
                str(num_dialogs),
                "delexicalize",
                str(cfg.delexicalize),
                "domain_setting",
                str(cfg.domain_setting),
            ]
        )
        + ".csv"
    )


def get_tokenizer(
    tokenizer_name: str = "gpt2", add_prefix_space: bool = False
) -> PreTrainedTokenizerFast:
    if tokenizer_name == "sentence-transformers/stsb-roberta-base-v2":
        add_prefix_space = True
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        pad_token=SpecialTokens.pad_token.value,
        bos_token=SpecialTokens.bos_token.value,
        eos_token=SpecialTokens.end_target.value,
        additional_special_tokens=SpecialTokens.list(),
        add_prefix_space=add_prefix_space,
    )
    return tokenizer


def get_token_id(tokenizer: AutoTokenizer, token_str: str) -> int:
    if tokenizer.name_or_path == "sentence-transformers/stsb-roberta-base-v2":
        return tokenizer(token_str)["input_ids"][1]
    return tokenizer(token_str)["input_ids"][0]


def get_text_in_between(
    text: str,
    start_token: str,
    end_token: str,
    default_value: any = None,
    multiple_values: bool = False,
) -> Union[str, list[str]]:
    if not multiple_values:
        try:
            idx1 = text.index(start_token)
            idx2 = text.index(end_token)
            res = text[idx1 + len(start_token) : idx2]
            # return res
            return res.strip()
        except ValueError:
            return default_value
    try:
        if SimpleTodConstants.NEW_LINES in text:
            text = text.replace(SimpleTodConstants.NEW_LINES, "")
        items = re.findall(f"{re.escape(start_token)}(.+?){re.escape(end_token)}", text)
        items = [item.strip() for item in items]
        if not items:
            return default_value
        return items
    except ValueError:
        return default_value


def remove_tokens_from_text(text: str, tokens: List[str]) -> str:
    for token in tokens:
        text = text.replace(token, "")
    return text
