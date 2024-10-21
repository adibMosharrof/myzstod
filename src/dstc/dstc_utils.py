import re
from pathlib import Path
from typing import List, Optional, Union, Dict
import numpy as np
from omegaconf import ListConfig
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizerFast,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
    GPTJForCausalLM,
    AutoModelWithLMHead,
    LlamaForCausalLM,
    T5ForConditionalGeneration,
)
import os
from multi_head.mh_model import GPT2MultiLMHeadModel

from my_enums import ZsTodConstants, SpecialTokens, Steps
import utils
from fuzzywuzzy import fuzz


def get_dstc_service_name(service_name: str) -> str:
    return service_name[: service_name.find("_")]


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
    clean_up_tokenization_spaces = False
    if "llama" in tokenizer_name:
        clean_up_tokenization_spaces = True
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        pad_token=SpecialTokens.pad_token.value,
        bos_token=SpecialTokens.bos_token.value,
        eos_token=SpecialTokens.end_target.value,
        additional_special_tokens=SpecialTokens.list(),
        add_prefix_space=add_prefix_space,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
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
    if not text:
        return default_value
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
        if ZsTodConstants.NEW_LINES in text:
            text = text.replace(ZsTodConstants.NEW_LINES, "")
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
    separator: str = ZsTodConstants.ITEM_SEPARATOR,
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


def get_model_class(model_name: str, is_mh_head: bool = False):
    if is_mh_head:
        return GPT2MultiLMHeadModel
    if model_name == "t5-base":
        return T5ForConditionalGeneration
    if "gpt-j" in model_name:
        return GPTJForCausalLM
    if "llama" in model_name:
        return LlamaForCausalLM
    if "t5" in model_name:
        return T5ForConditionalGeneration
    else:
        return AutoModelWithLMHead


def get_model_size(model: AutoModel) -> int:
    return sum(p.numel() for p in model.parameters())