from __future__ import annotations
import csv
import json
from collections import deque
from itertools import zip_longest
import os
from pathlib import Path
from typing import Tuple, Union

from dataclass_csv import DataclassReader
from omegaconf import DictConfig
import omegaconf
import wandb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configs.inference_config import InferenceConfig
    from configs.trainer_config import TrainerConfig

# from transformers.utils import logging
import logging


def get_logger(name: str = "transformers"):
    # logging.set_verbosity_info()
    # return logging.get_logger(name)
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
    # data = deque(iterable)
    # iterable.appendleft(None)
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def init_wandb(
    cfg: Union[InferenceConfig, TrainerConfig], omega_cfg: DictConfig, step: str
):
    wandb.config = omegaconf.OmegaConf.to_container(
        omega_cfg, resolve=True, throw_on_missing=True
    )
    out_dir = Path(os.getcwd())
    parent_without_year = "-".join(out_dir.parent.name.split("-")[1:])
    run_name = "/".join([parent_without_year, out_dir.name])
    group = "multi_head" if cfg.is_multi_head else "single_head"
    # num_dialogs = "_".join(map(str, cfg.num_dialogs))
    # tags = [cfg.model_name, num_dialogs, step]
    tags = [cfg.model_name, step]
    run = wandb.init(
        name=run_name,
        group=group,
        tags=tags,
        notes=cfg.wandb.notes if hasattr(cfg.wandb, "notes") else "",
        project=cfg.wandb.project,
        entity="adibm",
        settings=wandb.Settings(start_method="thread"),
    )
    wandb.log({"job_id": os.environ.get("SLURM_JOB_ID", "")})
