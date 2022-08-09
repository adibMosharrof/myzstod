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
from hydra_configs import TrainerConfig
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
        self.model_name = trainer_config.model_name
        self.pretrain_epochs = trainer_config.pretrain_epochs
        self.train_epochs = trainer_config.train_epochs
        self.train_batch_size = trainer_config.train_batch_size
        self.eval_batch_size = trainer_config.eval_batch_size
        self.test_batch_size = trainer_config.test_batch_size
        self.data_split_percent = trainer_config.data_split_percent
        self.eval_accumulation_steps = trainer_config.eval_accumulation_steps
        self.output_dir = Path(trainer_config.output_dir)
        self.logging_dir = trainer_config.logging_dir
        self.logging_steps = trainer_config.logging_steps
        self.max_token_len = trainer_config.max_token_len
        self.raw_data_root = trainer_config.raw_data_root
        self.data_prep_out_root = trainer_config.data_prep_out_root
        self.project_root = Path(trainer_config.project_root)
        self.num_workers = trainer_config.num_workers
        self.delexicalize = trainer_config.delexicalize
        self.num_dialogs = trainer_config.num_dialogs
        self.should_test = trainer_config.should_test
        self.generate_max_len = trainer_config.generate_max_len
        self.overwrite = trainer_config.overwrite
        self.domains = trainer_config.domains
        self.num_turns = trainer_config.num_turns
        self.pretrain_model_path = trainer_config.pretrain_model_path

    def run(self):

        self.tokenizer = dstc_utils.get_tokenizer(self.model_name)
        model = GPT2LMHeadModel.from_pretrained(self.model_name)
        model.resize_token_embeddings(len(self.tokenizer))
        model = model.cuda()

        dm = SimpleTodDataModule(
            tokenizer=self.tokenizer,
            data_prep_out_root=self.data_prep_out_root,
            raw_data_root=self.raw_data_root,
            project_root=self.project_root,
            out_root=self.data_prep_out_root,
            batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            test_batch_size=self.test_batch_size,
            data_split_percent=self.data_split_percent,
            max_token_len=self.max_token_len,
            num_workers=self.num_workers,
            delexicalize=self.delexicalize,
            num_dialogs=self.num_dialogs,
            domains=self.domains,
            num_turns=self.num_turns,
            overwrite=self.overwrite,
        )
        dm.setup()
        self.train(model, dm)
        print("Training done")
        print("-" * 80)
        if self.should_test:
            inf = Inference(
                model=model,
                project_root=self.project_root,
                dataloader=dm.test_dataloader(),
                num_workers=self.num_workers,
                data_prep_out_root=self.data_prep_out_root,
                data_split_percent=self.data_split_percent,
                eval_batch_size=self.eval_batch_size,
                test_batch_size=self.test_batch_size,
                max_token_len=self.max_token_len,
                raw_data_root=self.raw_data_root,
                delexicalize=self.delexicalize,
                num_test_dialogs=self.num_dialogs[2],
                generate_max_len=self.generate_max_len,
                domains=self.domains,
                num_turns=self.num_turns,
                tokenizer=self.tokenizer,
            )
            inf.test()

    def compute_metrics(self, preds: EvalPrediction):
        logits, labels = preds
        out_preds = np.argmax(logits, axis=-1)
        labels_txt = self.tokenizer.batch_decode(labels)
        predictions_txt = self.tokenizer.batch_decode(out_preds)
        bleu = evaluate.load("bleu")
        bleu_score = bleu.compute(predictions=predictions_txt, references=labels_txt)
        return {"bleu": bleu_score["bleu"]}

    def train(self, model: GPT2LMHeadModel, dm: SimpleTodDataModule):
        pretrain_out = str(self.output_dir / "pretrain")
        training_args = TrainingArguments(
            output_dir=pretrain_out,
            num_train_epochs=self.pretrain_epochs,
            logging_steps=self.logging_steps,
            load_best_model_at_end=True,
            save_strategy="epoch",
            save_total_limit=2,
            evaluation_strategy="epoch",
            eval_accumulation_steps=self.eval_accumulation_steps,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=self.logging_dir,
            dataloader_num_workers=self.num_workers,
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
        pre_trainer.pad_token_id = self.tokenizer.pad_token_id
        if not self.pretrain_model_path:
            pre_trainer.train()
            pre_trainer.save_model()
        else:
            pretrain_out = self.project_root / self.pretrain_model_path
        model_train = GPT2LMHeadModel.from_pretrained(pretrain_out)
        training_args.output_dir = str(self.output_dir / "train")
        training_args.num_train_epochs = self.train_epochs
        trainer = Trainer(
            model=model_train,
            args=training_args,
            train_dataset=dm.datasets["train"],
            eval_dataset=dm.datasets["dev"],
            data_collator=dm.training_collator,
        )
        trainer.pad_token_id = self.tokenizer.pad_token_id
        trainer.train()
        trainer.save_model()

        a = self.tokenizer.save_pretrained(self.output_dir)
        print("output_dir: ", os.getcwd())


class TodTrainer(Trainer):
    pad_token_id: int = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        # Shift so that tokens < n predict n
        input_ids = inputs["input_ids"]
        logits = model(input_ids, attention_mask=inputs["attention_mask"]).logits
        shift_labels = input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        # Calculate per-token loss
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return (loss, logits) if return_outputs else loss


@hydra.main(config_path="../config/trainer/", config_name="simple_tod_trainer")
def hydra_start(cfg: DictConfig) -> None:
    logging.set_verbosity_info()
    stt = SimpleTODTrainer(TrainerConfig(**cfg))
    stt.run()


if __name__ == "__main__":
    hydra_start()
