from omegaconf import DictConfig
import hydra
from transformers import (
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    logging,
)
from hydra_configs import DataModuleConfig, InferenceConfig, TrainerConfig
from inference import Inference
from my_datamodules import SimpleTodDataModule
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

        dm = SimpleTodDataModule(DataModuleConfig.from_trainer_config(self.cfg))
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
            train_dataset=dm.cfg.datasets["train"],
            eval_dataset=dm.cfg.datasets["dev"],
            data_collator=dm.pretraining_collator,
        )
        # pre_trainer.pad_token_id = self.cfg.tokenizer.pad_token_id
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
            train_dataset=dm.cfg.datasets["train"],
            eval_dataset=dm.cfg.datasets["dev"],
            data_collator=dm.training_collator,
        )
        # trainer.pad_token_id = self.cfg.tokenizer.pad_token_id
        trainer.train()
        trainer.save_model()

        self.cfg.tokenizer.save_pretrained(self.cfg.output_dir)
        print("output_dir: ", os.getcwd())


@hydra.main(config_path="../config/trainer/", config_name="simple_tod_trainer")
def hydra_start(cfg: DictConfig) -> None:
    stt = SimpleTODTrainer(TrainerConfig(**cfg))
    stt.run()


if __name__ == "__main__":
    hydra_start()
