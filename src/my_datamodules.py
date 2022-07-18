from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from dstc_dataclasses import Steps

import utils
from simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep
from simple_tod_dataclasses import SimpleTodConstants, SimpleTodTurnCsvRow
import dstc_utils


class SimpleTodDataModule(pl.LightningDataModule):
    steps = Steps.list()

    def __init__(
        self,
        num_workers=0,
        batch_size=32,
        eval_batch_size=32,
        test_batch_size=32,
        data_split_percent: List[float] = None,
        project_root: str = None,
        out_root: str = None,
        raw_data_root: str = None,
        data_prep_out_root: str = None,
        max_token_len: int = 128,
        num_dialogs: List[int] = None,
        preprocessing_model_name="simple_tod",
        dataset_name="dstc",
        model_name="gpt2",
        tokenizer=None,
        delexicalize: bool = True,
        overwrite: List[bool] = False,
        num_turns: int = 26,
        domains: List[str] = None,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.preprocessing_model_name = preprocessing_model_name
        self.project_root = Path(project_root)
        self.processed_data_root = self.project_root / data_prep_out_root
        self.raw_data_root = raw_data_root
        self.out_root = out_root
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.data_split_percent = data_split_percent
        self.max_token_len = max_token_len
        self.num_dialogs = num_dialogs
        self.dataset_name = dataset_name

        self.datasets: Dict[str, Dataset] = {}
        self.tokenizer = tokenizer
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False] * len(self.steps)
        self.num_turns = num_turns
        self.domains = domains or ["restaurant", "hotel", "attraction", "train"]

    def prepare_data(self):
        for step, num_dialog in zip(self.steps, self.num_dialogs):
            csv_file_path = dstc_utils.get_csv_data_path(
                step,
                num_dialog,
                self.delexicalize,
                self.processed_data_root,
                num_turns=self.num_turns,
                domains=self.domains,
            )
            if not csv_file_path.exists():
                stdp = SimpleTODDSTCDataPrep(
                    project_root=self.project_root,
                    data_root=self.raw_data_root,
                    out_root=self.out_root,
                    num_dialogs=self.num_dialogs,
                    domains=self.domains,
                    num_turns=self.num_turns,
                    overwrite=self.overwrite,
                    delexicalize=self.delexicalize,
                )
                stdp.run()

    def setup(self, stage: str = None):
        self.prepare_data()
        for step, split_percent, num_dialog in zip(
            self.steps, self.data_split_percent, self.num_dialogs
        ):

            csv_path = dstc_utils.get_csv_data_path(
                step,
                num_dialog,
                self.delexicalize,
                self.processed_data_root,
                num_turns=self.num_turns,
                domains=self.domains,
            )
            data = utils.read_csv_dataclass(csv_path, SimpleTodTurnCsvRow)
            data = data[: int(len(data) * split_percent)]
            self.datasets[step] = SimpleTodDataSet(
                data, tokenizer=self.tokenizer, max_token_len=self.max_token_len
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets[Steps.TRAIN],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.my_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[Steps.DEV],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.my_collate,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[Steps.TEST],
            batch_size=self.test_batch_size,
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

        return SimpleTodTestDataBatch(
            torch.stack([*contexts_tokens["input_ids"]]),
            torch.stack([*contexts_tokens["attention_mask"]]),
            torch.stack([*targets_tokens["input_ids"]]),
            torch.stack([*targets_tokens["attention_mask"]]),
            contexts,
            targets,
        )


class SimpleTodDataSet(Dataset):
    def __init__(
        self,
        data: List[SimpleTodTurnCsvRow],
        tokenizer: PreTrainedTokenizerFast = None,
        max_token_len: int = 128,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def tokenize(self, text: str):
        # tokens = self.tokenizer.encode_plus(
        # tokens = self.tokenizer.encode_plus(
        #     text,
        #     add_special_tokens=True,
        #     return_tensors="pt",
        #     truncation=True,
        #     padding="max_length",
        #     max_length=self.max_token_len,
        #     return_attention_mask=True,
        # )
        tokens = self.tokenizer(
            text, truncation=True, max_length=self.max_token_len, padding="max_length"
        )
        return {
            # "input_ids": tokens.input_ids.flatten(),
            # "attention_mask": tokens.attention_mask.flatten(),
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row: SimpleTodTurnCsvRow = self.data[idx]
        # return self.tokenize(row.context), self.tokenize(row.target)
        return row.context, row.target


@dataclass
class SimpleTodTestDataRow:
    context_tokens: torch.Tensor
    context_attention_masks: torch.Tensor
    label_tokens: torch.Tensor
    label_attention_masks: torch.Tensor
    contexts_text: str
    targets_text: str


@dataclass
class SimpleTodTestDataBatch:
    context_tokens: torch.Tensor
    context_attention_masks: torch.Tensor
    label_tokens: torch.Tensor
    label_attention_masks: torch.Tensor
    contexts_text: List[str]
    targets_text: List[str]

    def __iter__(self):
        for item in zip(
            self.context_tokens,
            self.context_attention_masks,
            self.label_tokens,
            self.label_attention_masks,
            self.contexts_text,
            self.targets_text,
        ):
            yield SimpleTodTestDataRow(*item)
