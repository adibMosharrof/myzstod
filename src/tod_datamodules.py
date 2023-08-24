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
            data.dialog_ids.append(item.dialog_id)
            data.turn_ids.append(item.turn_id)
            data.contexts.append(item.context)
            data.targets.append(item.target)
            data.schemas.append(item.schema)

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

            # context_tokens = self.train_tokenizer(
            #     # "".join([item.context, SpecialTokens.begin_target,SpecialTokens.begin_dsts, SpecialTokens.begin_dst])
            #     "".join([item.context, SpecialTokens.begin_target])
            # )[0]
            # # context_tokens = self.train_tokenizer(item.context)[0]
            # if self.cfg.is_multi_task:
            #     context_tokens = self._add_multi_task_prompt_token(context_tokens)

            # schema_tokens = self.train_tokenizer(item.schema)[0]
            # context_len = len(context_tokens)
            # schema_len = len(schema_tokens)
            # # handling case when input is greater than tokenizer length
            # unused_len = self.cfg.test_prompt_max_len - context_len - schema_len
            # if schema_len > self.cfg.test_prompt_max_len:
            #     raise ValueError("Schema is too long")
            # if unused_len < 0:
            #     context_start_tokens = context_tokens[:1]
            #     trimmed_context = context_tokens[unused_len * -1 + 1 :]
            #     context_tokens = torch.cat(
            #         [context_start_tokens, trimmed_context], axis=0
            #     )
            #     context_len = len(context_tokens)
            #     unused_len = 0
            #     print("unused length < 0")

            # pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
            # input_tokens = torch.cat([schema_tokens, context_tokens, pad])
            # data.input_ids.append(input_tokens)
            # attention_mask = torch.cat(
            #     [
            #         torch.full([schema_len + context_len], 1),
            #         torch.full([unused_len], 0),
            #     ]
            # )
            # data.attention_masks.append(attention_mask)

        return TodTestDataBatch(
            input_ids=torch.stack(data.input_ids),
            attention_masks=torch.stack(data.attention_masks),
            schemas_text=data.schemas,
            contexts_text=data.contexts,
            targets_text=data.targets,
            dialog_ids=data.dialog_ids,
            turn_ids=data.turn_ids,
        )
