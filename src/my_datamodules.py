from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
from dotmap import DotMap

import numpy as np
import pytorch_lightning as pl
from responses import target
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from my_enums import Steps

import utils
from simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep
from simple_tod_dataclasses import (
    SimpleTodTestDataBatch,
    SimpleTodTurnCsvRow,
)
import dstc_utils
from hydra_configs import DataModuleConfig, DataPrepConfig


class SimpleTodDataModule(pl.LightningDataModule):
    steps = Steps.list()
    _huggingface_ignore_label_id = -100

    def __init__(
        self,
        cfg: DataModuleConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.setup()

    def prepare_data(self):
        stdp = SimpleTODDSTCDataPrep(DataPrepConfig.from_dm_config(self.cfg))
        stdp.run()

    def setup(self):
        self.prepare_data()
        for step, split_percent, num_dialog in zip(
            self.steps, self.cfg.data_split_percent, self.cfg.num_dialogs
        ):
            csv_path = dstc_utils.get_csv_data_path(
                step,
                num_dialog,
                cfg=self.cfg,
            )
            try:
                data = utils.read_csv_dataclass(csv_path, SimpleTodTurnCsvRow)
                data = data[: int(len(data) * split_percent)]
            except FileNotFoundError:
                data = []
            self.cfg.datasets[step] = SimpleTodDataSet(data)

    def test_dataloader(self) -> Iterable[SimpleTodTestDataBatch]:
        return DataLoader(
            self.cfg.datasets[Steps.TEST],
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.my_test_collate,
            pin_memory=True,
        )

    def tokenize(self, item):
        return self.cfg.tokenizer(
            item,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.max_token_len,
        )

    def train_tokenizer(self, item):
        try:
            tokens = self.cfg.tokenizer.encode(
                item,
                return_tensors="pt",
            )
        except TypeError as e:
            tokens = torch.empty([1, 0], dtype=torch.int64)
        return tokens[0]

    def get_training_labels(self, context_len, unused_len, target_tokens):
        return torch.cat(
            [
                torch.full([context_len], self._huggingface_ignore_label_id),
                target_tokens,
                torch.full([unused_len], self._huggingface_ignore_label_id),
            ]
        )

    def pretraining_collator(self, batch: list[SimpleTodTurnCsvRow]):
        return self.training_collator(batch, True)

    def training_collator(
        self, batch: list[SimpleTodTurnCsvRow], is_pretrain: bool = False
    ):
        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            context_tokens = self.train_tokenizer(item.context)
            target_tokens = self.train_tokenizer(item.target)
            schema_tokens = self.train_tokenizer(item.schema)
            context_len = len(context_tokens)
            target_len = len(target_tokens)
            schema_len = len(schema_tokens)
            unused_len = self.cfg.max_token_len - context_len - target_len - schema_len
            start_token = torch.tensor([self.cfg.tokenizer.bos_token_id])
            end_token = torch.tensor([self.cfg.tokenizer.eos_token_id])
            # handling case when input is greater than tokenizer length
            if unused_len < 0:
                context_start_tokens = context_tokens[:1]
                trimmed_context = context_tokens[unused_len * -1 + 1 :]
                context_tokens = torch.cat(
                    [context_start_tokens, trimmed_context], axis=0
                )
                context_len = len(context_tokens)
                unused_len = 0

            pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
            input_tokens = torch.cat(
                [
                    start_token,
                    context_tokens,
                    schema_tokens,
                    target_tokens,
                    end_token,
                    pad,
                ]
            )
            if is_pretrain:
                label = input_tokens
            else:
                label = torch.cat(
                    [
                        torch.full(
                            [
                                context_len
                                + schema_len
                                + len(start_token)
                                + len(end_token)
                            ],
                            self._huggingface_ignore_label_id,
                        ),
                        target_tokens,
                        torch.full([unused_len], self._huggingface_ignore_label_id),
                    ]
                )
            attention_mask = torch.cat(
                [
                    torch.full(
                        [
                            context_len
                            + schema_len
                            + target_len
                            + len(start_token)
                            + len(end_token)
                        ],
                        1,
                    ),
                    torch.full([unused_len], 0),
                ]
            )
            input_ids.append(input_tokens)
            attention_masks.append(torch.tensor(attention_mask))
            labels.append(label)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }

    def my_test_collate(self, batch: list[SimpleTodTurnCsvRow]):
        data = DotMap(
            contexts_text=[],
            targets_text=[],
            schemas_text=[],
            dialog_ids=[],
            turn_ids=[],
            input_ids=[],
            attention_masks=[],
        )
        for item in batch:
            data.dialog_ids.append(item.dialog_id)
            data.turn_ids.append(item.turn_id)
            data.contexts_text.append(item.context)
            data.targets_text.append(item.target)
            data.schemas_text.append(item.schema)

            context_tokens = self.train_tokenizer(item.context)
            context_len = len(context_tokens)
            schema_tokens = self.train_tokenizer(item.schema)
            unused_len = self.cfg.context_max_len - context_len - len(schema_tokens)
            start_token = torch.tensor([self.cfg.tokenizer.bos_token_id])
            if unused_len < 0:
                context_start_tokens = context_tokens[:1]
                trimmed_context = context_tokens[unused_len * -1 + 1 :]
                context_tokens = torch.cat(
                    [context_start_tokens, trimmed_context], axis=0
                )

                unused_len = 0
                context_len = len(context_tokens)

            pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
            input_tokens = torch.cat([start_token, context_tokens, schema_tokens, pad])
            attention_mask = torch.cat(
                [
                    torch.full(
                        [len(start_token) + context_len + len(schema_tokens)],
                        1,
                    ),
                    torch.full([unused_len], 0),
                ]
            )
            data.input_ids.append(input_tokens)
            data.attention_masks.append(attention_mask)

        return SimpleTodTestDataBatch(
            turn_ids=data.turn_ids,
            dialog_ids=data.dialog_ids,
            contexts_text=data.contexts_text,
            targets_text=data.targets_text,
            schemas_text=data.schemas_text,
            input_ids=torch.stack(data.input_ids),
            attention_masks=torch.stack(data.attention_masks),
        )


class SimpleTodDataSet(Dataset):
    def __init__(
        self,
        data: List[SimpleTodTurnCsvRow],
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> SimpleTodTurnCsvRow:
        return self.data[idx]
