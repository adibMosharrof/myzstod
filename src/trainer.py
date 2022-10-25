from omegaconf import DictConfig
import hydra
from transformers import (
    AutoModel,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    logging,
)
from contrastive_dataclasses import ContrastiveTrainerHelper, ContrastiveTrainer
from hydra_configs import DataModuleConfig, InferenceConfig, TrainerConfig
from inference import Inference
from my_datamodules import TodDataModule
import os
import warnings

warnings.filterwarnings("ignore")


class SimpleTODTrainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
    ) -> None:
        self.cfg = trainer_config

    def run(self):

        model = GPT2LMHeadModel.from_pretrained(self.cfg.model_name)
        model.resize_token_embeddings(len(self.cfg.tokenizer))
        model = model.cuda()

        dm = TodDataModule(DataModuleConfig.from_trainer_config(self.cfg))
        self.train(model, dm)
        print("Training done")
        print("-" * 80)
        if self.cfg.should_test:
            inf = Inference(
                InferenceConfig.from_trainer_config(self.cfg, model),
            )
            inf.test()

    def _get_trainer(
        self,
        model_train: AutoModel,
        dm: TodDataModule,
        training_args: TrainingArguments,
    ) -> Trainer:
        if self.cfg.contrastive_model:
            trainer = ContrastiveTrainer(
                model=model_train,
                args=training_args,
                train_dataset=dm.cfg.datasets["train"],
                eval_dataset=dm.cfg.datasets["dev"],
                data_collator=dm.training_collator,
            )
            trainer.contrastive_helper = ContrastiveTrainerHelper(
                self.cfg.project_root / self.cfg.contrastive_model, self.cfg.tokenizer
            )
            return trainer
        trainer = Trainer(
            model=model_train,
            args=training_args,
            train_dataset=dm.cfg.datasets["train"],
            eval_dataset=dm.cfg.datasets["dev"],
            data_collator=dm.training_collator,
        )
        return trainer

    def train(self, model: GPT2LMHeadModel, dm: TodDataModule):
        pretrain_out = str(self.cfg.out_dir / "pretrain")
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
            train_dataset=dm.cfg.datasets["train"],
            eval_dataset=dm.cfg.datasets["dev"],
            data_collator=dm.pretraining_collator,
        )
        if not self.cfg.pretrain_model_path:
            pre_trainer.train()
            pre_trainer.save_model()
        else:
            pretrain_out = self.cfg.project_root / self.cfg.pretrain_model_path
        model_train = GPT2LMHeadModel.from_pretrained(pretrain_out)
        training_args.output_dir = str(self.cfg.out_dir / "train")
        training_args.num_train_epochs = self.cfg.train_epochs
        trainer = self._get_trainer(model_train, dm, training_args)
        trainer.train()
        trainer.save_model()

        self.cfg.tokenizer.save_pretrained(self.cfg.out_dir)
        print("output_dir: ", os.getcwd())


@hydra.main(config_path="../config/trainer/", config_name="simple_tod_trainer")
def hydra_start(cfg: DictConfig) -> None:
    stt = SimpleTODTrainer(TrainerConfig(**cfg))
    stt.run()


if __name__ == "__main__":
    hydra_start()
