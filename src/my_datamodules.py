from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
from responses import target
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from dstc_dataclasses import Steps

import utils
from simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep
from simple_tod_dataclasses import (
    SimpleTodConstants,
    SimpleTodDatasetItem,
    SimpleTodTestDataBatch,
    SimpleTodTurnCsvRow,
    SpecialTokens,
    get_multi_task_special_tokens,
)
import dstc_utils


class SimpleTodDataModule(pl.LightningDataModule):
    steps = Steps.list()
    _huggingface_ignore_label_id = -100

    def __init__(
        self,
        num_workers=8,
        batch_size=32,
        eval_batch_size=32,
        test_batch_size=32,
        data_split_percent: List[float] = None,
        project_root: str = None,
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        data_prep_out_root: str = "processed_data/simple_tod",
        max_token_len: int = 128,
        num_dialogs: List[int] = None,
        preprocessing_model_name="simple_tod",
        dataset_name="dstc",
        model_name="gpt2",
        tokenizer=None,
        delexicalize: bool = False,
        overwrite: List[bool] = None,
        num_turns: int = 26,
        domains: List[str] = None,
        is_multi_task: bool = False,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.preprocessing_model_name = preprocessing_model_name
        self.project_root = Path(project_root)
        self.processed_data_root = self.project_root / data_prep_out_root
        self.raw_data_root = raw_data_root
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.data_split_percent = data_split_percent
        self.max_token_len = max_token_len
        self.num_dialogs = num_dialogs
        self.dataset_name = dataset_name

        self.datasets: Dict[str, Dataset] = {}
        self.tokenizer = tokenizer or dstc_utils.get_tokenizer()
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False] * len(self.steps)
        self.num_turns = num_turns
        self.domains = domains or ["restaurant", "hotel", "attraction", "train"]
        self.is_multi_task = is_multi_task

    def prepare_data(self):
        stdp = SimpleTODDSTCDataPrep(
            project_root=self.project_root,
            data_root=self.raw_data_root,
            out_root=self.processed_data_root,
            num_dialogs=self.num_dialogs,
            domains=self.domains,
            num_turns=self.num_turns,
            overwrite=self.overwrite,
            delexicalize=self.delexicalize,
            is_multi_task=self.is_multi_task,
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
                is_multi_task=self.is_multi_task,
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
            collate_fn=self.training_collator,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[Steps.DEV],
            batch_size=self.eval_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.training_collator,
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

    def test_tokenize(self, item):
        return self.tokenizer(
            item,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.test_max_token_len,
        )

    def train_eval_tokenize(self, item):
        return self.tokenizer.encode(
            item,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_token_len,
        )

    def training_collator(self, batch: list[SimpleTodDatasetItem]):
        if self.is_multi_task:
            batch = self.multi_task_preprocessor(batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            context_tokens = self.train_eval_tokenize(item.context)[0]
            target_tokens = self.train_eval_tokenize(item.target)[0]
            context_len = len(context_tokens)
            target_len = len(target_tokens)
            unused_len = self.max_token_len - context_len - target_len
            # handling case when input is greater than tokenizer length
            if unused_len < 0:
                context_start_token = context_tokens[0]
                context_tokens = context_tokens[unused_len * -1 :]
                context_len = len(context_tokens)
                unused_len = 0

            pad = torch.full([unused_len], self.tokenizer.pad_token_id)
            input_tokens = torch.cat([context_tokens, target_tokens, pad])
            label = torch.cat(
                [
                    torch.full([context_len], self._huggingface_ignore_label_id),
                    target_tokens,
                    torch.full([unused_len], self._huggingface_ignore_label_id),
                ]
            )
            attention_mask = torch.cat(
                [torch.full([context_len + target_len], 1), torch.full([unused_len], 0)]
            )
            input_ids.append(input_tokens)
            attention_masks.append(torch.tensor(attention_mask))
            labels.append(label)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }

    def pretraining_collator(self, batch: list[SimpleTodDatasetItem]):
        if self.is_multi_task:
            batch = self.multi_task_preprocessor(batch)
        texts = [x.context + x.target for x in batch]
        targets = [x.target for x in batch]
        texts_tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_token_len,
        )
        targets_tokens = self.tokenize(targets)
        input_ids = torch.stack([*texts_tokens["input_ids"]])
        labels = torch.stack([*targets_tokens["input_ids"]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.stack([*texts_tokens["attention_mask"]]),
            "labels": input_ids,
        }

    def my_test_collate(self, batch):
        contexts = [x.context for x in batch]
        targets = [x.target for x in batch]
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

    def _extract_from_target(self, target, start_token, end_token):
        try:
            start_index = target.index(start_token)
            end_index = target.index(end_token)
        except ValueError:
            raise ValueError(
                f"could not find start or end token in target, {start_token}, {end_token}"
            )
        return target[start_index : end_index + len(end_token)]

    def multi_task_preprocessor(
        self, batch: list[SimpleTodDatasetItem]
    ) -> list[SimpleTodDatasetItem]:
        out = []
        multi_task_special_tokens = get_multi_task_special_tokens()
        for item in batch:
            for mtst in multi_task_special_tokens:
                try:
                    text = self._extract_from_target(
                        item.target, mtst.start_token, mtst.end_token
                    )
                except ValueError:
                    continue
                row = SimpleTodDatasetItem(
                    context=item.context + mtst.prompt_token,
                    target=text,
                )
                out.append(row)
        return out


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
        return SimpleTodDatasetItem(row.context, row.target)
