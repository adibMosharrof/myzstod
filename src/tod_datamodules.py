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

    def tokenizer_text(self, text):
        if not text:
            return torch.empty([1, 0], dtype=torch.int32)
        tok_out = self.cfg.tokenizer(
            text,
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
        for item in batch:
            input_tokens, label, attention_mask = self.collate_single_item(
                item.context,
                item.schema,
                item.target,
                self.cfg.max_token_len,
                is_pretrain,
            )
            input_ids.append(input_tokens)
            attention_masks.append(attention_mask)
            labels.append(label)
            targets_text.append(item.target)

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }

        if not is_pretrain and self.cfg.contrast_with and self.cfg.is_multi_task:
            out["mt_prompt_token_ids"] = torch.tensor(mt_prompt_ids)

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
        for item in batch:
            data.dialog_ids.append(self.tokenizer_text(item.dialog_id))
            data.turn_ids.append(self.tokenizer_text(item.turn_id))
            data.contexts.append(self.tokenizer_text(item.context))
            data.targets.append(self.tokenizer_text(item.target))
            # data.schemas.append(item.schema)

            input_tokens, _, attention_mask = self.collate_single_item(
                "".join(
                    [
                        item.context,
                        # SpecialTokens.begin_target,
                        # SpecialTokens.begin_dsts,
                        # SpecialTokens.begin_dst,
                    ]
                ),
                item.schema,
                "",
                self.cfg.test_prompt_max_len,
                True,
            )
            data.input_ids.append(input_tokens)
            data.attention_masks.append(attention_mask)

        return TodTestDataBatch(
            input_ids=torch.stack(data.input_ids),
            attention_masks=torch.stack(data.attention_masks),
            contexts_text=torch.stack(data.contexts),
            targets_text=torch.stack(data.targets),
            dialog_ids=torch.stack(data.dialog_ids),
            turn_ids=torch.stack(data.turn_ids),
        )
