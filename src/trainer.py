# import pandas as pd
from pathlib import Path
from typing import List
from omegaconf import DictConfig
import torch
import hydra
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    EvalPrediction,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    logging,
)
import evaluate
from dstc_dataclasses import Steps
from hydra_configs import InferenceConfig, TrainerConfig
from inference import Inference
from my_datamodules import SimpleTodDataModule
from simple_tod_dataclasses import SpecialTokens, TokenizerTokens
import os
import dstc_utils
from torch.nn import CrossEntropyLoss
import warnings

warnings.filterwarnings("ignore")


class SimpleTODTrainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
    ) -> None:
        self.cfg = trainer_config
        # self.model_name = trainer_config.model_name
        # self.pretrain_epochs = trainer_config.pretrain_epochs
        # self.train_epochs = trainer_config.train_epochs
        # self.train_batch_size = trainer_config.train_batch_size
        # self.eval_batch_size = trainer_config.eval_batch_size
        # self.test_batch_size = trainer_config.test_batch_size
        # self.data_split_percent = trainer_config.data_split_percent
        # self.eval_accumulation_steps = trainer_config.eval_accumulation_steps
        # self.output_dir = Path(trainer_config.output_dir)
        # self.logging_dir = trainer_config.logging_dir
        # self.logging_steps = trainer_config.logging_steps
        # self.max_token_len = trainer_config.max_token_len
        # self.raw_data_root = trainer_config.raw_data_root
        # self.data_prep_out_root = trainer_config.data_prep_out_root
        # self.project_root = Path(trainer_config.project_root)
        # self.num_workers = trainer_config.num_workers
        # self.delexicalize = trainer_config.delexicalize
        # self.num_dialogs = trainer_config.num_dialogs
        # self.should_test = trainer_config.should_test
        # self.generate_max_len = trainer_config.generate_max_len
        # self.overwrite = trainer_config.overwrite
        # self.domains = trainer_config.domains
        # self.num_turns = trainer_config.num_turns
        # self.pretrain_model_path = trainer_config.pretrain_model_path

    def run(self):

        model = GPT2LMHeadModel.from_pretrained(self.cfg.model_name)
        model.resize_token_embeddings(len(self.cfg.tokenizer))
        model = model.cuda()

        dm = SimpleTodDataModule(
            tokenizer=self.cfg.tokenizer,
            data_prep_out_root=self.cfg.data_prep_out_root,
            raw_data_root=self.cfg.raw_data_root,
            project_root=self.cfg.project_root,
            batch_size=self.cfg.train_batch_size,
            eval_batch_size=self.cfg.eval_batch_size,
            test_batch_size=self.cfg.test_batch_size,
            data_split_percent=self.cfg.data_split_percent,
            max_token_len=self.cfg.max_token_len,
            num_workers=self.cfg.num_workers,
            delexicalize=self.cfg.delexicalize,
            num_dialogs=self.cfg.num_dialogs,
            domains=self.cfg.domains,
            num_turns=self.cfg.num_turns,
            overwrite=self.cfg.overwrite,
            is_multi_task=self.cfg.is_multi_task,
        )
        dm.setup()
        self.train(model, dm)
        print("Training done")
        print("-" * 80)
        if self.cfg.should_test:
            inf = Inference(
                InferenceConfig(
                    model=model,
                    project_root=self.cfg.project_root,
                    num_workers=self.cfg.num_workers,
                    data_prep_out_root=self.cfg.data_prep_out_root,
                    data_split_percent=self.cfg.data_split_percent,
                    eval_batch_size=self.cfg.eval_batch_size,
                    test_batch_size=self.cfg.test_batch_size,
                    max_token_len=self.cfg.max_token_len,
                    raw_data_root=self.cfg.raw_data_root,
                    delexicalize=self.cfg.delexicalize,
                    num_test_dialogs=self.cfg.num_dialogs[2],
                    generate_max_len=self.cfg.generate_max_len,
                    domains=self.cfg.domains,
                    num_turns=self.cfg.num_turns,
                    tokenizer=self.cfg.tokenizer,
                )
            )
            inf.test()

    def train(self, model: GPT2LMHeadModel, dm: SimpleTodDataModule):
        pretrain_out = str(self.cfg.output_dir / "pretrain")
        training_args = TrainingArguments(
            output_dir=pretrain_out,
            num_train_epochs=self.cfg.pretrain_epochs,
            logging_steps=self.cfg.logging_steps,
            load_best_model_at_end=True,
            save_strategy="epoch",
            save_total_limit=2,
            evaluation_strategy="epoch",
            eval_accumulation_steps=self.cfg.eval_accumulation_steps,
            per_device_train_batch_size=self.cfg.train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=self.cfg.logging_dir,
            dataloader_num_workers=self.cfg.num_workers,
            dataloader_pin_memory=True,
        )

        # start training
        pre_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dm.datasets["train"],
            eval_dataset=dm.datasets["dev"],
            data_collator=dm.pretraining_collator,
        )
        pre_trainer.pad_token_id = self.cfg.tokenizer.pad_token_id
        if not self.cfg.pretrain_model_path:
            pre_trainer.train()
            pre_trainer.save_model()
        else:
            pretrain_out = self.cfg.project_root / self.cfg.pretrain_model_path
        model_train = GPT2LMHeadModel.from_pretrained(pretrain_out)
        training_args.output_dir = str(self.cfg.output_dir / "train")
        training_args.num_train_epochs = self.cfg.train_epochs
        trainer = Trainer(
            model=model_train,
            args=training_args,
            train_dataset=dm.datasets["train"],
            eval_dataset=dm.datasets["dev"],
            data_collator=dm.training_collator,
        )
        trainer.pad_token_id = self.cfg.tokenizer.pad_token_id
        trainer.train()
        trainer.save_model()

        self.cfg.tokenizer.save_pretrained(self.cfg.output_dir)
        print("output_dir: ", os.getcwd())


@hydra.main(config_path="../config/trainer/", config_name="simple_tod_trainer")
def hydra_start(cfg: DictConfig) -> None:
    logging.set_verbosity_info()
    stt = SimpleTODTrainer(TrainerConfig(**cfg))
    stt.run()


if __name__ == "__main__":
    hydra_start()
