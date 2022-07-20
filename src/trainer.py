# import pandas as pd
from pathlib import Path
from typing import List
import hydra
from omegaconf import DictConfig
import torch
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
)
import evaluate
from dstc_dataclasses import Steps
from inference import Inference
from my_datamodules import SimpleTodDataModule
from simple_tod_dataclasses import SpecialTokens, TokenizerTokens
import os
import dstc_utils
from torch.nn import CrossEntropyLoss


class SimpleTODTrainer:
    def __init__(
        self,
        model_name: str = None,
        epochs: int = 2,
        train_batch_size: int = 30,
        eval_batch_size: int = 30,
        test_batch_size: int = 30,
        eval_accumulation_steps: int = 10,
        data_split_percent: list[float] = None,
        raw_data_root: str = None,
        output_dir: str = "results",
        logging_dir: str = "logs",
        logging_steps: int = 10,
        max_token_len: int = 128,
        data_prep_out_root: str = None,
        project_root: str = None,
        num_workers: int = 0,
        delexicalize: bool = True,
        num_dialogs: List[int] = None,
        model_checkpoint_path: str = None,
        should_test: bool = False,
        generate_max_len: int = 200,
        domains: List[str] = None,
        num_turns: int = 26,
        overwrite: List[bool] = None,
    ) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.data_split_percent = data_split_percent
        self.eval_accumulation_steps = eval_accumulation_steps
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.max_token_len = max_token_len
        self.raw_data_root = raw_data_root
        self.data_prep_out_root = data_prep_out_root
        self.project_root = Path(project_root)
        self.num_workers = num_workers
        self.delexicalize = delexicalize
        self.num_dialogs = num_dialogs
        self.model_checkpoint_path = model_checkpoint_path
        self.should_test = should_test
        self.generate_max_len = generate_max_len
        self.overwrite = overwrite or [False, False, False]
        self.domains = domains or ["restaurant", "hotel", "attraction"]
        self.num_turns = num_turns

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

    # made changes here, check next time to see if it works
    def train(self, model, dm):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
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
        # trainer = Trainer(
        trainer = TodTrainer(
            model=model,
            args=training_args,
            train_dataset=dm.datasets["train"],
            eval_dataset=dm.datasets["dev"],
            # compute_metrics=self.compute_metrics,
            data_collator=dm.my_collate,
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
    stt = SimpleTODTrainer(
        epochs=cfg.epochs,
        model_name=cfg.model_name,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        test_batch_size=cfg.test_batch_size,
        output_dir=cfg.output_dir,
        logging_dir=cfg.logging_dir,
        logging_steps=cfg.logging_steps,
        max_token_len=cfg.max_token_len,
        raw_data_root=cfg.raw_data_root,
        data_prep_out_root=cfg.data_prep_out_root,
        project_root=cfg.project_root,
        num_workers=cfg.num_workers,
        data_split_percent=cfg.data_split_percent,
        delexicalize=cfg.delexicalize,
        num_dialogs=cfg.num_dialogs,
        should_test=cfg.should_test,
        domains=cfg.domains,
        num_turns=cfg.num_turns,
        overwrite=cfg.overwrite,
        generate_max_len=cfg.generate_max_len,
    )
    stt.run()


if __name__ == "__main__":
    hydra_start()
