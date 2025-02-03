from dotmap import DotMap
from simple_tod_dataclasses import NlgTestDataBatch
from tod.turns.zs_tod_turn import TodTurnApiCallCsvRow, TodTurnCsvRow
import torch
from abc import ABC, abstractmethod
from utilities.tokenizer_utilities import TokenizerUtilities
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from base_datamodule import TodTrainRowCollator


class BaseCollator(ABC):

    _huggingface_ignore_label_id = -100

    def __init__(
        self,
        tokenizer,
        nlg_prompt_cls,
        max_token_len: int = 1024,
        test_prompt_max_len: int = 750,
    ):
        self.tokenizer = tokenizer
        self.nlg_prompt_cls = nlg_prompt_cls
        self.test_prompt_max_len = test_prompt_max_len
        self.max_token_len = max_token_len

    @abstractmethod
    def prepare_item(
        self,
        context_tokens,
        target_tokens,
        pad_tokens,
        target_max_len,
        is_test: bool,
    ) -> "TodTrainRowCollator":
        pass

    def tod_train_collate(self, batch: list[TodTurnCsvRow]):
        all_input_tokens = []
        all_labels = []
        all_attention_masks = []

        target_max_len = self.max_token_len - self.test_prompt_max_len
        for item in batch:
            row = self.collate_single_item(item, target_max_len)
            all_input_tokens.append(row.input_tokens)
            all_attention_masks.append(row.attention_mask)
            all_labels.append(row.label)
        return {
            "input_ids": torch.stack(all_input_tokens),
            "labels": torch.stack(all_labels),
            "attention_mask": torch.stack(all_attention_masks),
        }

    def tod_test_collate(self, batch: list[TodTurnApiCallCsvRow]):
        data = self.get_test_data_cols()
        max_lengths = DotMap(
            domains=100,
        )
        target_max_len = self.max_token_len - self.test_prompt_max_len
        for item in batch:
            row = self.collate_single_item(item, target_max_len, is_test=True)
            self.update_test_data(data, row, max_lengths, target_max_len, item)
        return NlgTestDataBatch(
            dialog_ids=torch.stack(data.dialog_ids),
            turn_ids=torch.stack(data.turn_ids),
            input_ids=torch.stack(data.input_tokens),
            attention_masks=torch.stack(data.attention_masks),
            labels=torch.stack(data.labels),
            turn_row_types=torch.stack(data.turn_row_types),
            is_retrievals=torch.stack(data.is_retrievals),
            is_slot_fills=torch.stack(data.is_slot_fills),
            is_multi_domain_api_calls=torch.stack(data.is_multi_domain_api_calls),
            domain_ids=torch.stack(data.domain_ids),
            is_single_domains=torch.stack(data.is_single_domains),
            current_user_utterance_tokens=torch.stack(
                data.current_user_utterance_tokens
            ),
        )

    def get_test_data_cols(self):
        data = DotMap(
            {
                key: []
                for key in [
                    "input_tokens",
                    "attention_masks",
                    "dialog_ids",
                    "turn_ids",
                    "labels",
                    "turn_row_types",
                    "is_retrievals",
                    "is_slot_fills",
                    "is_multi_domain_api_calls",
                    "domain_ids",
                    "is_single_domains",
                    "current_user_utterance_tokens",
                ]
            }
        )

        return data

    def update_test_data(self, data, row, max_lengths, target_max_len, item):
        data.input_tokens.append(row.input_tokens)
        data.attention_masks.append(row.attention_mask)
        data.labels.append(row.label)
        data.turn_row_types.append(torch.tensor(item.turn_row_type))
        data.is_retrievals.append(torch.tensor(item.is_retrieval))
        data.is_slot_fills.append(torch.tensor(item.is_slot_fill))
        data.turn_ids.append(torch.tensor(int(item.turn_id)))
        data.dialog_ids.append(torch.tensor(int(item.dialog_id)))
        multi_dom = getattr(item, "is_multi_domain_api_call", 0)
        if not multi_dom:
            multi_dom = 0
        data.is_multi_domain_api_calls.append(torch.tensor(int(multi_dom)))
        domain_tokens = TokenizerUtilities.tokenize_with_pad(
            text=item.domains_original,
            max_len=max_lengths.domains,
            tokenizer=self.tokenizer,
        )
        data.domain_ids.append(domain_tokens["input_ids"][0])
        current_user_utterance_tokens = TokenizerUtilities.tokenize_with_pad(
            text=item.current_user_utterance,
            tokenizer=self.tokenizer,
        )
        data.is_single_domains.append(torch.tensor(item.is_single_domain))
        data.current_user_utterance_tokens.append(
            current_user_utterance_tokens["input_ids"][0]
        )

    def collate_single_item(
        self, item: TodTurnCsvRow, target_max_len: int, is_test: bool = False
    ) -> "TodTrainRowCollator":
        context_text = self.get_context_text(item)
        context_tokens, context_unused_len = self.get_context_tokens_and_unused_len(
            item, context_text
        )
        pad = torch.full([context_unused_len], self.tokenizer.pad_token_id)
        target_tokens = self.get_target_tokens(item.target, target_max_len)

        return self.prepare_item(
            context_tokens=context_tokens,
            target_tokens=target_tokens,
            pad_tokens=pad,
            target_max_len=target_max_len,
            is_test=is_test,
        )

    def get_context_text(self, item):
        context_text = self.nlg_prompt_cls.get_prompt(
            domain=item.domains,
            schema=item.schema,
            dialog_history=item.context,
            domains_original=item.domains_original,
        )
        return context_text

    def get_context_tokens_and_unused_len(self, item, context_text):
        context_tokens = TokenizerUtilities.tokenize(
            tokenizer=self.tokenizer, text=context_text
        )
        context_unused_len = self.test_prompt_max_len - len(context_tokens)
        if context_unused_len < 0:
            context_tokens = self.trim_dialog_history(item, -context_unused_len, 0)
            context_unused_len = self.test_prompt_max_len - len(context_tokens)
        return context_tokens, context_unused_len

    def trim_dialog_history(
        self,
        item: TodTurnCsvRow,
        trim_len: int,
        counter=0,
    ):
        dialog_history_tokens = TokenizerUtilities.tokenize(
            tokenizer=self.tokenizer,
            text=item.context,
        )
        trimmed_history_tokens = dialog_history_tokens[trim_len + 15 :]
        trimmed_history_text = self.tokenizer.decode(trimmed_history_tokens)
        context_text = self.nlg_prompt_cls.get_prompt(
            item.domains,
            item.schema,
            trimmed_history_text,
            item.domains_original,
        )
        context_tokens = TokenizerUtilities.tokenize(
            tokenizer=self.tokenizer, text=context_text
        )
        if counter > 100:
            a = 1
        if len(context_tokens) > self.test_prompt_max_len:
            overflow_tokens = len(context_tokens) - self.test_prompt_max_len
            return self.trim_dialog_history(item, -overflow_tokens, counter + 1)
        return context_tokens

    def get_target_tokens(self, text: str, target_max_len: int):
        target_tokens = TokenizerUtilities.tokenize(text=text, tokenizer=self.tokenizer)
        if len(target_tokens) > target_max_len:
            raise Exception("Target is too long")
        return target_tokens
