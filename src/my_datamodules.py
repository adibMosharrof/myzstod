from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import utils
from simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep
from simple_tod_dataclasses import SimpleTodTurnCsvRow


class SimpleTodDataModule(pl.LightningDataModule):
    steps = ["train", "dev", "test"]

    def __init__(
        self,
        num_workers=0,
        batch_size=32,
        data_root=None,
        max_token_len=256,
        num_dialogs=2,
        preprocessing_model_name="simple_tod",
        dataset_name="dstc",
        model_name="gpt2",
        tokenizer=None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.preprocessing_model_name = preprocessing_model_name
        self.data_root = Path(data_root) / self.preprocessing_model_name
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.num_dialogs = num_dialogs
        self.dataset_name = dataset_name

        self.datasets: Dict[str, Dataset] = {}
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer

    def prepare_data(self):
        for step in self.steps:
            step_dir = self.data_root / step
            csv_file = (
                step_dir
                / f"{self.preprocessing_model_name}_{self.dataset_name}_{self.num_dialogs}.csv"
            )
            # csv_file_paths.append(csv_file)
            if not csv_file.exists():
                stdp = SimpleTODDSTCDataPrep(
                    self.data_root,
                    self.data_root / self.preprocessing_model_name,
                    self.num_dialogs,
                )
                stdp.run()

    def setup(self, stage: str = None):
        for step in self.steps:
            step_dir = self.data_root / step
            csv_path = (
                step_dir
                / f"{self.preprocessing_model_name}_{self.dataset_name}_{self.num_dialogs}.csv"
            )
            data = utils.read_csv_dataclass(csv_path, SimpleTodTurnCsvRow)
            self.datasets[step] = SimpleTodDataSet(
                data, tokenizer=self.tokenizer, max_token_len=self.max_token_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.my_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["dev"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.my_collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.my_test_collate,
            pin_memory=True,
        )

    def tokenize(self, item):
        return self.tokenizer(
            item,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
        )

    def my_collate(self, batch):
        texts = [x[0] + x[1] for x in batch]
        targets = [x[1] for x in batch]
        texts_tokens = self.tokenize(texts)
        targets_tokens = self.tokenize(targets)
        input_ids = torch.stack([*texts_tokens["input_ids"]])
        labels = torch.stack([*targets_tokens["input_ids"]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.stack([*texts_tokens["attention_mask"]]),
            "labels": input_ids,
        }

    def my_test_collate(self, batch):
        contexts = [x[0] for x in batch]
        targets = [x[1] for x in batch]
        contexts_tokens, targets_tokens = self.tokenize(contexts), self.tokenize(
            targets
        )

        return {
            "input_ids": torch.stack([*contexts_tokens["input_ids"]]),
            "attention_mask": torch.stack([*contexts_tokens["attention_mask"]]),
            "labels": torch.stack([*targets_tokens["input_ids"]]),
            "contexts_text": contexts,
            "targets_text": targets,
        }
        # return {
        #     "context_input_ids": torch.stack([*contexts_tokens["input_ids"]]),
        #     "context_attention_mask": torch.stack([*contexts_tokens["attention_mask"]]),
        #     "target_input_ids": torch.stack([*targets_tokens["input_ids"]]),
        #     "target_attention_mask": torch.stack([*targets_tokens["attention_mask"]]),
        # }


class SimpleTodDataSet(Dataset):
    def __init__(
        self,
        data: List[SimpleTodTurnCsvRow],
        tokenizer: PreTrainedTokenizerFast = None,
        max_token_len: int = 256,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def tokenize(self, text: str):
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
            return_attention_mask=True,
        )
        return {
            "input_ids": tokens.input_ids.flatten(),
            "attention_mask": tokens.attention_mask.flatten(),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row: SimpleTodTurnCsvRow = self.data[idx]
        # return self.tokenize(row.context), self.tokenize(row.target)
        return row.context, row.target
