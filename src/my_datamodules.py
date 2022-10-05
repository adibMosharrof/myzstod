from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pytorch_lightning as pl
import torch
from responses import target
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import dstc_utils
import utils
from hydra_configs import DataModuleConfig, DataPrepConfig
from my_enums import Steps
from simple_tod_dataclasses import SimpleTodTestDataBatch, SimpleTodTurnCsvRow
from simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep


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
        return tokens

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
            context_tokens = self.train_tokenizer(item.context)[0]
            target_tokens = self.train_tokenizer(item.target)[0]
            schema_tokens = self.train_tokenizer(item.schema)[0]
            context_len = len(context_tokens)
            target_len = len(target_tokens)
            schema_len = len(schema_tokens)
            unused_len = self.cfg.max_token_len - context_len - target_len - schema_len
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
                [context_tokens, schema_tokens, target_tokens, pad]
            )
            if is_pretrain:
                label = input_tokens
            else:
                label = torch.cat(
                    [
                        torch.full(
                            [context_len + schema_len],
                            self._huggingface_ignore_label_id,
                        ),
                        target_tokens,
                        torch.full([unused_len], self._huggingface_ignore_label_id),
                    ]
                )
            attention_mask = torch.cat(
                [
                    torch.full([context_len + schema_len + target_len], 1),
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
        dialog_ids, turn_ids, contexts, schemas, targets = [], [], [], [], []
        for item in batch:
            dialog_ids.append(item.dialog_id)
            turn_ids.append(item.turn_id)
            contexts.append(item.context)
            targets.append(item.target)

        # contexts_tokens, targets_tokens = self.tokenize(contexts), self.tokenize(
        #     targets
        # )
        contexts_tokens = self.tokenize(contexts)

        return SimpleTodTestDataBatch(
            input_ids=torch.stack([*contexts_tokens["input_ids"]]),
            attention_masks=torch.stack([*contexts_tokens["attention_mask"]]),
            # torch.stack([*targets_tokens["input_ids"]]),
            # torch.stack([*targets_tokens["attention_mask"]]),
            schemas_text=schemas,
            contexts_text=contexts,
            targets_text=targets,
            dialog_ids=dialog_ids,
            turn_ids=turn_ids,
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
