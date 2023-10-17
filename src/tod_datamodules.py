from typing import Optional
from dotmap import DotMap

import torch
from base_datamodule import BaseDataModule
from configs.dm_config import DataModuleConfig

from my_enums import (
    DstcSystemActions,
    MultiTaskNames,
    ZsTodActionAttributes,
    SpecialTokens,
    Steps,
)
from simple_tod_dataclasses import TodTestDataBatch
import dstc.dstc_utils as dstc_utils
from torch.utils.data import DataLoader, Dataset
import random

from tod.turns.zs_tod_turn import TodTurnCsvRow


class TodDataModule(BaseDataModule):
    _huggingface_ignore_label_id = -100

    def __init__(
        self,
        cfg: DataModuleConfig,
        steps: list[Steps] = None,
        tod_turn_row_cls=TodTurnCsvRow,
        task_name: Optional[MultiTaskNames] = None,
    ):
        super().__init__(
            cfg, steps, tod_turn_row_cls=tod_turn_row_cls, task_name=task_name
        )

    def tokenizer_text(self, text, max_len=None):
        if not text:
            return torch.empty([1, 0], dtype=torch.int)
        tok_out = self.cfg.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return tok_out["input_ids"][0]

    def training_collator(self, batch: list[TodTurnCsvRow], is_pretrain: bool = False):
        input_ids = []
        attention_masks = []
        labels = []
        targets_text = []
        mt_prompt_ids = []
        all_special_tokens_target_mask = []
        all_special_tokens_vocab_mask = []
        for item in batch:
            if "t5" in self.cfg.model_name:
                row = self.t5_collate_single_item(
                    item,
                    self.cfg.max_token_len,
                )
            else:
                row = self.collate_single_item(
                    item,
                    self.cfg.max_token_len,
                    is_pretrain,
                )
            (
                input_tokens,
                label,
                attention_mask,
            ) = (
                row.input_tokens,
                row.label,
                row.attention_mask,
            )
            if self.cfg.is_scale_grad:
                special_tokens_target_mask, special_tokens_vocab_mask = (
                    row.special_tokens_target_mask,
                    row.special_tokens_vocab_mask,
                )
            input_ids.append(input_tokens)
            attention_masks.append(attention_mask)
            labels.append(label)
            targets_text.append(item.target)
            if self.cfg.is_scale_grad:
                all_special_tokens_target_mask.append(special_tokens_target_mask)
                all_special_tokens_vocab_mask.append(special_tokens_vocab_mask)

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }

        if not is_pretrain and self.cfg.contrast_with and self.cfg.is_multi_task:
            out["mt_prompt_token_ids"] = torch.tensor(mt_prompt_ids)
        if self.cfg.is_scale_grad:
            out["special_tokens_target_mask"] = torch.stack(
                all_special_tokens_target_mask
            )
            out["special_tokens_vocab_mask"] = torch.stack(
                all_special_tokens_vocab_mask
            )
        return out

    def my_test_collate(self, batch: list[TodTurnCsvRow]) -> TodTestDataBatch:
        data = DotMap(
            {
                key: []
                for key in [
                    "input_ids",
                    "attention_masks",
                    "dialog_ids",
                    "turn_ids",
                    "contexts",
                    "schemas",
                    "targets",
                ]
            }
        )
        max_lengths = DotMap(
            dialog_id=10,
            turn_id=10,
            context=self.cfg.test_prompt_max_len,
            target=self.cfg.max_token_len - self.cfg.test_prompt_max_len,
        )
        for item in batch:
            data.dialog_ids.append(
                self.tokenizer_text(item.dialog_id, max_lengths.dialog_id)
            )
            data.turn_ids.append(self.tokenizer_text(item.turn_id, max_lengths.turn_id))
            data.contexts.append(self.tokenizer_text(item.context, max_lengths.context))
            data.targets.append(self.tokenizer_text(item.target, max_lengths.target))
            # data.schemas.append(item.schema)

            row = self.collate_single_item(
                item,
                self.cfg.test_prompt_max_len,
                True,
            )
            data.input_ids.append(row.input_tokens)
            data.attention_masks.append(row.attention_mask)

        return TodTestDataBatch(
            input_ids=torch.stack(data.input_ids),
            attention_masks=torch.stack(data.attention_masks),
            contexts_text=torch.stack(data.contexts),
            targets_text=torch.stack(data.targets),
            dialog_ids=torch.stack(data.dialog_ids),
            turn_ids=torch.stack(data.turn_ids),
        )
