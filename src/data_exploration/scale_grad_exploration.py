import os
import sys

import torch


sys.path.insert(0, os.path.abspath("./src"))

from tod.turns.zs_tod_turn import TodTurnScaleGradCsvRow

from scale_grad.scale_grad_model import ScaleGradModel

from pathlib import Path
from dotmap import DotMap

from tod.turns.scalegrad_turn_csv_row import ScaleGradTurnCsvRow
from my_enums import Steps
from tod_datamodules import TodDataModule
from configs.dm_config import DataModuleConfig
from torch.utils.data import DataLoader
import utils
from peft import prepare_model_for_kbit_training, PeftModelForCausalLM
from accelerate import Accelerator


class ScaleGradExploration:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.tokenizer = utils.get_tokenizer(cfg.tokenizer, add_prefix_space=True)
        self.epochs = 1

    def get_data(self):
        # model = ScaleGradModel.from_pretrained(self.cfg.model_name)
        scale_grad_gamma = 0.4
        accelerator = Accelerator()
        device_map = {"": torch.cuda.current_device()}
        model = ScaleGradModel.from_pretrained(
            self.cfg.model_name,
            {"gamma": scale_grad_gamma},
            device_map=device_map,
        )
        model.resize_token_embeddings(len(self.cfg.tokenizer))
        model = prepare_model_for_kbit_training(model, self.cfg)
        config = utils.get_lora_config(self.cfg.model_name)
        model = PeftModelForCausalLM(model, config)
        model.print_trainable_parameters()

        dm = TodDataModule(
            DataModuleConfig(**self.cfg),
            steps=Steps.list(),
            tod_turn_row_cls=TodTurnScaleGradCsvRow,
        )
        dl = DataLoader(
            dm.datasets[Steps.TRAIN], batch_size=2, collate_fn=dm.training_collator
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
        model.train()
        for epoch in range(self.epochs):
            for data in iter(dl):
                optimizer.zero_grad()
                out = model(**data)
                print(f"train loss {out.loss}")
                accelerator.backward(out.loss)
                optimizer.step()
                a = 1


if __name__ == "__main__":
    sg = ScaleGradExploration(
        DotMap(
            raw_data_root=Path("data/dstc8-schema-guided-dialogue/"),
            data_prep_out_root="data/processed_data/",
            project_root=Path("/mounts/u-amo-d1/adibm-data/projects/ZSToD"),
            # num_dialogs=[127, 20, 34],
            num_dialogs=[1, 1, 1],
            data_split_percent=[1, 1, 1],
            overwrite=[0, 0, 0],
            max_token_len=1024,
            test_prompt_max_len=750,
            # model_name="huggyllama/llama-7b",
            # tokenizer="adibm/sgd-llama-tokenizer",
            tokenizer="adibm/sgd-opt-tokenizer",
            model_name="facebook/opt-350m",
            # out_file_path="data_exploration/token_lengths",
            should_add_schema=True,
            should_add_sys_actions=False,
            should_add_user_actions=False,
            should_add_service_results=True,
            is_scale_grad=True,
            train_domain_settings=["seen"],
            dev_domain_settings=["all"],
            test_domain_settings=[["all"]],
        )
    )
    sg.get_data()
