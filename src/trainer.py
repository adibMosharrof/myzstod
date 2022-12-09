from collections import defaultdict
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig
import hydra
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    logging,
    EarlyStoppingCallback,
    IntervalStrategy,
    GPT2Config,
)
from contrastive import Contrastive
from contrastive_dataclasses import ContrastiveTrainerHelper, ContrastiveTrainer
from hydra_configs import (
    ContrastiveConfig,
    DataModuleConfig,
    TrainerConfig,
    InferenceConfig,
)
from inference import Inference
from my_datamodules import TodDataModule
import os
import warnings
import my_enums
import dstc_utils
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
# os.environ["NCCL_DEBUG"] = "INFO"


class SimpleTODTrainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
    ) -> None:
        self.cfg = trainer_config

    def print_cuda_info(self, step=""):
        if step:
            print(f"Step: {step}")
        print(torch.cuda.memory_allocated() / 1024**2)
        print(torch.cuda.memory_cached() / 1024**2)

    def run(self):
        self.print_cuda_info("init")
        # model = GPT2LMHeadModel.from_pretrained(self.cfg.model_name)
        # model.resize_token_embeddings(len(self.cfg.tokenizer))
        # heads_to_prune = defaultdict(list)
        # for layer in range(12):
        #     for head in range(12):
        #         if head < self.cfg.n_head and layer < self.cfg.n_layer:
        #             continue
        #         heads_to_prune[layer].append(head)
        # model.prune_heads(heads_to_prune)
        # model = model.cuda()
        current_dir = os.getcwd()
        dm = TodDataModule(DataModuleConfig.from_trainer_config(self.cfg))
        # self.cfg.tokenizer = dstc_utils.get_trained_tokenizer(self.cfg)
        if self.cfg.train_model_path:
            pretrained_model_path = str(self.cfg.project_root/self.cfg.train_model_path)
        else:
            pretrained_model_path = self.pretrain_model(dm)
        self.print_cuda_info("after pretrain")
        self._setup_contrastive()
        self.print_cuda_info("contrastive model created")
        torch.cuda.empty_cache()
        self.print_cuda_info("empty cache before training")
        out_dir = self.train_model(pretrained_model_path, dm)            
        full_out_dir = str(Path(current_dir) / out_dir)
        self.print_cuda_info("after train")
        # out_dir = self.train(model, dm)
        print("Training done")
        print("-" * 80)
        if self.cfg.should_test:
            inf = Inference(
                # InferenceConfig.from_trainer_config(self.cfg, model),
                InferenceConfig.from_trainer_config(self.cfg, full_out_dir),
            )
            inf.test()
        print(full_out_dir)

    def _setup_contrastive(self) -> Optional[AutoTokenizer]:
        if not self.cfg.contrast_with:
            return None
        if self.cfg.contrastive_model:
            model_or_path = SentenceTransformer(
                self.cfg.project_root / self.cfg.contrastive_model
            ).cuda()
        else:
            c = Contrastive(ContrastiveConfig.from_trainer_config(self.cfg))
            model_or_path = c.run()
        self.contrastive_helper = ContrastiveTrainerHelper(
            model_or_path,
            self.cfg.tokenizer,
            self.cfg.contrastive_max_token_len,
            Contrastive.get_start_end_tokens(self.cfg.contrast_with),
            self.cfg.is_multi_task,
            self.cfg.ce_loss_weight,
            self.cfg.contrastive_loss_weight
        )
        return self.contrastive_helper.contrastive_model.tokenizer

    def _get_trainer(
        self,
        model_train: AutoModel,
        dm: TodDataModule,
        training_args: TrainingArguments,
    ) -> Trainer:
        if self.cfg.contrast_with:
            trainer = ContrastiveTrainer(
                model=model_train,
                args=training_args,
                train_dataset=dm.cfg.datasets["train"],
                eval_dataset=dm.cfg.datasets["dev"],
                data_collator=dm.training_collator,
                callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                )
            ],
            )
            trainer.contrastive_helper = self.contrastive_helper
            return trainer
        trainer = Trainer(
            model=model_train,
            args=training_args,
            train_dataset=dm.cfg.datasets["train"],
            eval_dataset=dm.cfg.datasets["dev"],
            data_collator=dm.training_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                )
            ],
        )
        return trainer

    def _get_training_args(
        self, step_name: str, epochs: int, train_batch_size: int
    ) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(self.cfg.out_dir / step_name),
            num_train_epochs=epochs,
            logging_steps=self.cfg.logging_steps,
            load_best_model_at_end=True,
            save_strategy=IntervalStrategy.STEPS,
            save_total_limit=5,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.cfg.eval_steps,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            metric_for_best_model="eval_loss",
            eval_accumulation_steps=self.cfg.eval_accumulation_steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=self.cfg.logging_dir,
            dataloader_num_workers=self.cfg.num_workers,
            dataloader_pin_memory=True,
            report_to="all",
        )

    def pretrain_model(self, dm: TodDataModule) -> str:
        if self.cfg.pretrain_model_path:
            path = self.cfg.project_root / self.cfg.pretrain_model_path
            if path.exists():
                return str(path)
        training_args = self._get_training_args(
            "pretrain", self.cfg.pretrain_epochs, self.cfg.pretrain_batch_size
        )
        model = GPT2LMHeadModel.from_pretrained(self.cfg.model_name)
        model.resize_token_embeddings(len(self.cfg.tokenizer))
        pre_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dm.cfg.datasets["train"],
            eval_dataset=dm.cfg.datasets["dev"],
            data_collator=dm.pretraining_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                )
            ],
        )
        pre_trainer.train()
        pre_trainer.save_model()
        # return str(self.cfg.project_root/training_args.output_dir)
        return training_args.output_dir

    def train_model(self, path, dm) -> str:
        model = GPT2LMHeadModel.from_pretrained(path)
        training_args = self._get_training_args(
            "train", self.cfg.train_epochs, self.cfg.train_batch_size
        )
        trainer = self._get_trainer(model, dm, training_args)
        trainer.train()
        trainer.save_model()
        out_dir = os.getcwd()
        print("training output_dir: ", out_dir)
        return training_args.output_dir

    # not used anymore
    def train(self, model: GPT2LMHeadModel, dm: TodDataModule):
        pretrain_out = str(self.cfg.out_dir / "pretrain")
        training_args = TrainingArguments(
            output_dir=pretrain_out,
            num_train_epochs=self.cfg.pretrain_epochs,
            logging_steps=self.cfg.logging_steps,
            load_best_model_at_end=True,
            save_strategy=IntervalStrategy.STEPS,
            save_total_limit=5,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.cfg.eval_steps,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            metric_for_best_model="eval_loss",
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
            train_dataset=dm.cfg.datasets["train"],
            eval_dataset=dm.cfg.datasets["dev"],
            data_collator=dm.pretraining_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                )
            ],
        )
        if not self.cfg.pretrain_model_path:
            pre_trainer.train()
            pre_trainer.save_model()
        else:
            pretrain_out = self.cfg.project_root / self.cfg.pretrain_model_path
        # model_train = GPT2LMHeadModel.from_pretrained(pretrain_out)
        self.print_cuda_info("pretrained model created")
        training_args.output_dir = str(self.cfg.out_dir / "train")
        training_args.num_train_epochs = self.cfg.train_epochs
        # trainer = self._get_trainer(model_train, dm, training_args)
        trainer = self._get_trainer(model, dm, training_args)
        # torch.cuda.empty_cache()
        trainer.train()
        trainer.save_model()
        self.print_cuda_info("trained model created")
        # self.cfg.tokenizer.save_pretrained(self.cfg.out_dir)
        out_dir = os.getcwd()
        print("output_dir: ", out_dir)
        return out_dir


@hydra.main(config_path="../config/trainer/", config_name="simple_tod_trainer")
def hydra_start(cfg: DictConfig) -> None:
    stt = SimpleTODTrainer(TrainerConfig(**cfg))
    stt.run()


if __name__ == "__main__":
    hydra_start()
