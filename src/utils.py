from __future__ import annotations
import csv
import json
from collections import deque
from itertools import zip_longest
import os
from pathlib import Path
from typing import Optional, Tuple, Union

from dataclass_csv import DataclassReader
from omegaconf import DictConfig
import omegaconf
from omegaconf.listconfig import ListConfig
import torch
import wandb
from typing import TYPE_CHECKING
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from transformers.trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from configs.inference_config import InferenceConfig
    from configs.trainer_config import TrainerConfig
    from configs.task_arithmetic_config import TaskArithmeticConfig

# from transformers.utils import logging
import logging
import peft
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
)
from accelerate import Accelerator

import csv
from itertools import zip_longest
import json
import logging
import os
from pathlib import Path
import re
from typing import Tuple, Union, Optional
from dataclass_csv import DataclassReader
from omegaconf import DictConfig, ListConfig
import omegaconf
import wandb
from fuzzywuzzy import fuzz

from my_enums import SpecialTokens, ZsTodConstants


def get_dialog_file_paths(data_root, step):
    pattern = "dialogues"
    files = sorted(os.listdir(data_root / step))
    file_paths = [data_root / step / f for f in files if pattern in f]
    return file_paths


def get_csv_data_path(
    step: str = "train",
    num_dialogs: int = 1,
    cfg: any = None,
    data_root: Optional[Path] = None,
):
    sgdx_versions = ["v1", "v2", "v3", "v4", "v5"]
    version = "v0"
    if cfg.raw_data_root.stem in sgdx_versions:
        version = cfg.raw_data_root.stem
    domain_setting = get_domain_setting_str(cfg.domain_setting)
    base = data_root if data_root else cfg.processed_data_root
    step_dir = base / step
    return step_dir / (
        "_".join(
            [
                version,
                "context_type",
                cfg.context_type,
                # "multi_head",
                # str(cfg.is_multi_head),
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
                "domain_setting",
                get_domain_setting_str(domain_setting),
                "train_domains",
                str(cfg.train_domain_percentage),
            ]
        )
        + ".csv"
    )


def get_domain_setting_str(domain_setting: Union[list[str], ListConfig, str]):
    if isinstance(domain_setting, (list, ListConfig)):
        return "_".join(domain_setting)
    return domain_setting


def get_logger(name: str = "transformers"):
    return logging.getLogger(__name__)


def append_csv(data, file_name: Path):
    with open(file_name, "a", encoding="UTF8", newline="") as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerows(data)


def write_csv(headers: list[str], data, file_name: Path):
    with open(file_name, "w", encoding="UTF8", newline="") as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        csvwriter.writerow(headers)
        csvwriter.writerows(data)


def write_json(data: list[any], path: str):
    with open(path, "w") as f:
        json.dump(data, f)


def read_json(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read_csv(path: str) -> Tuple[list[list[str]], list[str]]:
    fields = []
    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        fields = next(reader)
        for r in reader:
            rows.append(r)
    return rows, fields


def read_csv_dataclass(path: str, d_class):
    with open(path) as f:
        reader = DataclassReader(f, d_class)
        return [r for r in reader]


def get_num_items(num, max_value):
    if num == None:
        return max_value
    return num


def read_lines_in_file(path: Path) -> list[any]:
    with open(path) as file:
        lines = [line.rstrip() for line in file]
    return lines

def grouper(iterable, n=2, fillvalue=None):
    # "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def init_wandb(cfg: any, cmd_args: any, step: str, entity="None"):
    out_dir = Path(os.getcwd())
    parent_without_year = "-".join(out_dir.parent.name.split("-")[1:])
    gpu_name = f'gpu:{cmd_args.local_rank}'
    run_name = "/".join([parent_without_year, out_dir.name])
    tags = [cfg.wandb.task, cfg.model_name, step]
    run = wandb.init(
        # name=gpu_name,
        group=run_name,
        tags=tags,
        notes=cfg.wandb.notes if hasattr(cfg.wandb, "notes") else "",
        project=cfg.wandb.project,
        # entity=entity,
        # settings=wandb.Settings(start_method="thread"),
    )
    wandb.log({"job_id": os.environ.get("SLURM_JOB_ID", "")})


def fuzzy_string_match(ref: str, hyp: str) -> float:
    return fuzz.token_set_ratio(ref, hyp) / 100.0


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


def get_tokenizer(
    tokenizer_name: str = "gpt2",
    add_prefix_space: bool = False,
    tokenizer_path="tokenizer",
) -> AutoTokenizer:
    tok_path = Path(tokenizer_path)
    if tok_path.exists():
        return AutoTokenizer.from_pretrained(tok_path)
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


def get_model_class(model_name: str):
    if model_name == "t5-base":
        return T5ForConditionalGeneration
    return GPT2LMHeadModel


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


def remove_tokens_from_text(text: str, tokens: list[str]) -> str:
    for token in tokens:
        text = text.replace(token, "")
    return text


def init_wandb(
    cfg: Union[InferenceConfig, TrainerConfig, TaskArithmeticConfig],
    omega_cfg: DictConfig,
    step: str,
):
    wandb.config = omegaconf.OmegaConf.to_container(
        omega_cfg, resolve=True, throw_on_missing=True
    )
    out_dir = Path(os.getcwd())
    parent_without_year = "-".join(out_dir.parent.name.split("-")[1:])
    run_name = "/".join([parent_without_year, out_dir.name])
    # group = "multi_head" if cfg.is_multi_head else "single_head"
    # num_dialogs = "_".join(map(str, cfg.num_dialogs))
    # tags = [cfg.model_name, num_dialogs, step]
    tags = [cfg.wandb.task, cfg.model_name, step]
    run = wandb.init(
        name=run_name,
        # group=group,
        tags=tags,
        notes=cfg.wandb.notes if hasattr(cfg.wandb, "notes") else "",
        project=cfg.wandb.project,
        entity="adibm",
        settings=wandb.Settings(start_method="thread"),
    )
    wandb.log({"job_id": os.environ.get("SLURM_JOB_ID", "")})


def get_8bit_model(model_name: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_name, load_in_8bit=True, device_map="auto", torch_dtype=torch.bfloat16
    )


def get_4bit_model(model_name: str) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
    )
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map,
    )
    model.config.pretraining_tp = 1
    return model


def load_quantized_model(path: Path, tokenizer: AutoTokenizer, quantization_dtype=8):
    config = PeftConfig.from_pretrained(path)
    if quantization_dtype == 8:
        model = get_8bit_model(config.base_model_name_or_path)
    elif quantization_dtype == 4:
        model = get_4bit_model(config.base_model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, path)
    return model


def get_modules_to_save(model_name: str):
    if "gpt-j" in model_name:
        return ["lm_head", "wte"]
    return ["lm_head", "embed_tokens"]


def get_lora_config(model_name: str) -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        base_model_name_or_path=model_name,
        modules_to_save=get_modules_to_save(model_name),
    )


class PeftSavingCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(state.best_model_checkpoint, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(
            state.best_model_checkpoint, "pytorch_model.bin"
        )
        os.remove(pytorch_model_path) if os.path.exists(pytorch_model_path) else None
