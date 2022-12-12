import re
from pathlib import Path
from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os

from my_enums import SimpleTodConstants, SpecialTokens, Steps
import utils
from fuzzywuzzy import fuzz


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
    cfg: any = None,
    data_root: Optional[Path] = None,
    domain_setting: Optional[str] = None,
):
    sgdx_versions = ["v1", "v2", "v3", "v4", "v5"]
    version = "v0"
    if cfg.raw_data_root.stem in sgdx_versions:
        version = cfg.raw_data_root.stem
    dom_sett = domain_setting if domain_setting else str(cfg.domain_setting)
    base = data_root if data_root else cfg.processed_data_root
    step_dir = base / step
    return step_dir / (
        "_".join(
            [
                version,
                "context_type",
                cfg.context_type,
                "multi_head",
                str(cfg.is_multi_head),
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
                dom_sett.upper(),
                "train_domains",
                str(cfg.train_domain_percentage),
            ]
        )
        + ".csv"
    )


def get_corpus(cfg) -> List[str]:
    # root = cfg.project_root / cfg.data_prep_out_root / Steps.DEV.value
    # file = root / os.listdir(root)[0]
    file = get_csv_data_path(
        Steps.TRAIN.value,
        cfg.num_dialogs[0],
        cfg=cfg,
        data_root=cfg.project_root / cfg.data_prep_out_root,
        domain_setting=cfg.train_domain_setting,
    )
    csv_file = pd.read_csv(file, names=["context", "target"])
    for _, row in csv_file.iterrows():
        yield row["context"] + " " + row["target"]


def get_trained_tokenizer(cfg, save_path: str = "tokenizer") -> PreTrainedTokenizerFast:
    tokenizer = get_tokenizer(cfg.tokenizer_name)
    # new_tok = tokenizer.train_new_from_iterator(get_corpus(cfg), 52000, new_special_tokens=SpecialTokens.list())
    new_tok = tokenizer.train_new_from_iterator(get_corpus(cfg), 52000)
    new_tok.save_pretrained(save_path)
    return new_tok


def get_tokenizer(
    tokenizer_name: str = "gpt2",
    add_prefix_space: bool = False,
    tokenizer_path="tokenizer",
) -> PreTrainedTokenizerFast:
    print("************getting tokenizer*************")
    tok_path = Path(tokenizer_path)
    if tok_path.exists():
        return AutoTokenizer.from_pretrained(tok_path)
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
    tokenizer.save_pretrained(tokenizer_path)
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
            return res
            # return res.strip()
        except ValueError:
            return default_value
    try:
        if SimpleTodConstants.NEW_LINES in text:
            text = text.replace(SimpleTodConstants.NEW_LINES, "")
        items = re.findall(f"{re.escape(start_token)}(.+?){re.escape(end_token)}", text)
        items = [item for item in items]
        if not items:
            return default_value
        return items
    except ValueError:
        return default_value


def extract_section_and_split_items_from_text(
    text: str,
    start_token: str,
    end_token: str,
    separator: str = SimpleTodConstants.ITEM_SEPARATOR,
    default_value: any = [],
    multiple_values: bool = False,
) -> np.ndarray:
    section_txts = get_text_in_between(
        text, start_token, end_token, default_value, multiple_values=multiple_values
    )
    if not section_txts:
        return default_value
    if type(section_txts) == list:
        out = [st.split(separator) for st in section_txts]
        return np.concatenate(out, axis=0, dtype=str)
    return np.array(section_txts.split(separator), dtype=str)


def remove_tokens_from_text(text: str, tokens: List[str]) -> str:
    for token in tokens:
        text = text.replace(token, "")
    return text


def get_slot_value_match_score(
    ref: Union[str, list[str]], hyp: Union[str, list[str]], is_categorical: bool
) -> float:
    if not ref and not hyp:
        return 1.0
    if isinstance(ref, str):
        ref = [ref]
    if isinstance(hyp, str):
        hyp = [hyp]
    score = 0.0
    for ref_item in ref:
        for hyp_item in hyp:
            if is_categorical:
                match_score = float(ref_item == hyp_item)
            else:
                match_score = fuzzy_string_match(ref_item, hyp_item)
            score = max(score, match_score)
    return score


def fuzzy_string_match(ref: str, hyp: str) -> float:
    return fuzz.token_set_ratio(ref, hyp) / 100.0
