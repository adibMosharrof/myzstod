from dotmap import DotMap

import torch
from sentence_transformers import InputExample
from base_datamodule import BaseDataModule
from contrastive_dataclasses import ContrastiveUtils

from hydra_configs import DataModuleConfig
from my_enums import (
    DstcSystemActions,
    SimpleTodActionAttributes,
    SimpleTodConstants,
    SpecialTokens,
    Steps,
)
from simple_tod_dataclasses import SimpleTodAction, TodTestDataBatch, TodTurnCsvRow
import dstc_utils
from torch.utils.data import DataLoader, Dataset
import random


class TodDataModule(BaseDataModule):
    steps = Steps.list()
    _huggingface_ignore_label_id = -100

    def __init__(
        self,
        cfg: DataModuleConfig,
    ):
        super().__init__(cfg)

    def training_collator(self, batch: list[TodTurnCsvRow], is_pretrain: bool = False):
        input_ids = []
        attention_masks = []
        labels = []
        targets_text = []
        contrast_tokens = []
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
                if self.cfg.contrast_with:
                    contrast_tokens.append(
                        self._get_contrast_tokens(item.target, self.cfg.contrast_with)
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
            targets_text.append(item.target)

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }
        if not is_pretrain and self.cfg.contrast_with:
            out["contrastive_tokens"] = torch.stack(contrast_tokens)

        return out

    def my_test_collate(self, batch: list[TodTurnCsvRow]) -> TodTestDataBatch:
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

        return TodTestDataBatch(
            input_ids=torch.stack(data.input_ids),
            attention_masks=torch.stack(data.attention_masks),
            schemas_text=data.schemas,
            contexts_text=data.contexts,
            targets_text=data.targets,
            dialog_ids=data.dialog_ids,
            turn_ids=data.turn_ids,
        )

    def _get_contrast_tokens(self, target: str, contrast_with: str) -> list[int]:
        (
            start_token,
            end_token,
            multiple_values,
        ) = ContrastiveUtils._get_tokens_from_contrast_with(contrast_with)
        text = dstc_utils.get_text_in_between(
            target, start_token, end_token, multiple_values=multiple_values
        )
        if isinstance(text, list):
            text = SimpleTodConstants.ITEM_SEPARATOR.join(text)
        tokens = self.contrastive_tokenizer(text)
        return tokens
