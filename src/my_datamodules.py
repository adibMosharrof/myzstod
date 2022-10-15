from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
from dotmap import DotMap

import numpy as np
import pytorch_lightning as pl
import torch
from responses import target
from torch.utils.data import DataLoader, Dataset, default_collate
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import dstc_utils
import utils
from hydra_configs import DataModuleConfig, DataPrepConfig
from my_enums import SpecialTokens, Steps
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
        self.prompt_token_map = {}

    def _get_token_id(self, text: str) -> int:
        return self.cfg.tokenizer.encode(text)[0]

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

    def test_dataloader(self) -> SimpleTodTestDataBatch:
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
        contexts_text = []
        labels_text = []
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
                [schema_tokens, context_tokens, target_tokens, pad]
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

            # contexts_text.append(
            #     self.cfg.tokenizer.decode(input_tokens[attention_mask > 0])
            # )
            # labels_text.append(
            # self.cfg.tokenizer.decode(
            #     input_tokens[label != self._huggingface_ignore_label_id]
            # )
            # )
        # text_csv_data = np.column_stack([contexts_text, labels_text])
        # path = f"text_csv_data_{is_pretrain }.csv"
        # utils.append_csv(text_csv_data, path)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }

    def my_test_collate(
        self, batch: list[SimpleTodTurnCsvRow]
    ) -> SimpleTodTestDataBatch:

        data = DotMap(
            dict.fromkeys(
                [
                    "input_ids",
                    "attention_masks",
                    "dialog_ids",
                    "turn_ids",
                    "contexts",
                    "schemas",
                    "targets",
                ],
                [],
            )
        )
        for item in batch:
            data.dialog_ids.append(item.dialog_id)
            data.turn_ids.append(item.turn_id)
            data.contexts.append(item.context)
            data.targets.append(item.target)
            data.schemas.append(item.schema)
            context_tokens = self.train_tokenizer(
                item.context + SpecialTokens.begin_target
            )[0]
            # context_tokens = self.train_tokenizer(item.context)[0]
            if self.cfg.is_multi_task:
                context_tokens = self._add_multi_task_prompt_token(context_tokens)

            schema_tokens = self.train_tokenizer(item.schema)[0]
            context_len = len(context_tokens)
            schema_len = len(schema_tokens)
            # handling case when input is greater than tokenizer length
            unused_len = self.cfg.test_prompt_max_len - context_len - schema_len
            if schema_len > self.cfg.test_prompt_max_len:
                raise ValueError("Schema is too long")
            if unused_len < 0:
                context_start_tokens = context_tokens[:1]
                trimmed_context = context_tokens[unused_len * -1 + 1 :]
                context_tokens = torch.cat(
                    [context_start_tokens, trimmed_context], axis=0
                )
                context_len = len(context_tokens)
                unused_len = 0

            pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
            input_tokens = torch.cat([schema_tokens, context_tokens, pad])
            data.input_ids.append(input_tokens)
            attention_mask = torch.cat(
                [
                    torch.full([schema_len + context_len], 1),
                    torch.full([unused_len], 0),
                ]
            )
            data.attention_masks.append(attention_mask)

        return SimpleTodTestDataBatch(
            input_ids=torch.stack(data.input_ids),
            attention_masks=torch.stack(data.attention_masks),
            schemas_text=data.schemas,
            contexts_text=data.contexts,
            targets_text=data.targets,
            dialog_ids=data.dialog_ids,
            turn_ids=data.turn_ids,
        )

    def _get_filler_token_from_prompt(self, prompt_token: int, prompt_token_map: dict):
        try:
            filler_token = prompt_token_map[int(prompt_token)]
        except KeyError:
            raise ValueError("Prompt token not found")
        return filler_token

    def _add_multi_task_prompt_token(
        self, context_tokens: torch.Tensor
    ) -> torch.Tensor:
        if not self.prompt_token_map.keys():
            self.prompt_token_map = {
                self._get_token_id(SpecialTokens.prompt_dst): self._get_token_id(
                    SpecialTokens.begin_dsts
                ),
                self._get_token_id(SpecialTokens.prompt_action): self._get_token_id(
                    SpecialTokens.begin_action
                ),
                self._get_token_id(SpecialTokens.prompt_response): self._get_token_id(
                    SpecialTokens.begin_response
                ),
            }
        prompt_token = self._get_filler_token_from_prompt(
            context_tokens[-2], self.prompt_token_map
        )
        out = torch.cat([context_tokens, torch.tensor([prompt_token])])
        return out


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
