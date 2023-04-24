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
from transformers import AutoModelForCausalLM
from accelerate import Accelerator


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


def load_quantized_model(path: Path, tokenizer):
    current_device = Accelerator().process_index
    config = PeftConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        # return_dict=True,
        load_in_8bit=True,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    # model = model.cuda()
    # model = PeftModel.from_pretrained(model, path, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, path)
    return model
    key_list = [
        key for key, _ in model.base_model.model.named_modules() if "lora" not in key
    ]
    for key in key_list:
        parent, target, target_name = model.base_model._get_submodules(key)
        if isinstance(target, peft.tuners.lora.Linear):
            bias = target.bias is not None
            new_module = torch.nn.Linear(
                target.in_features, target.out_features, bias=bias
            )
            model.base_model._replace_module(parent, target_name, new_module, target)

    model = model.base_model.model
    return model
    row = tokenizer(
        "<|begincontext|>I am looking to eat somewhere<|endcontext|>",
        return_tensors="pt",
    )
    output_tokens = model.generate(
        inputs=row["input_ids"].cuda(),
        attention_mask=row["attention_mask"].cuda(),
        max_length=150,
    )

    print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=False))
    a = 1


class PeftSavingCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(state.best_model_checkpoint, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(
            state.best_model_checkpoint, "pytorch_model.bin"
        )
        os.remove(pytorch_model_path) if os.path.exists(pytorch_model_path) else None
