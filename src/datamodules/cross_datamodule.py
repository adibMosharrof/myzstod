from dotmap import DotMap
from base_datamodule import TodTrainCrossRowCollator, TodTrainRowCollator
from my_enums import Steps
from prompts.nlg_prompt_manager import NlgPromptFactory
from simple_tod_dataclasses import CrossTestDataBatch
from tod.turns.zs_tod_turn import (
    TodTurnApiCallCsvRow,
    TodTurnCsvRow,
    TodTurnCsvRowFactory,
)
from tod_datamodules import TodDataModule
from configs.dm_config import DataModuleConfig
import torch
from utilities.tokenizer_utilities import TokenizerUtilities
from accelerate import Accelerator


class CrossDataModule(TodDataModule):
    def __init__(self, cfg, tokenizer, schemas, encoder_model):
        super().__init__(
            DataModuleConfig(tokenizer=tokenizer, **cfg),
            steps=Steps.list(),
            tod_turn_row_cls=TodTurnCsvRowFactory.get_handler(cfg),
        )
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.schemas = schemas
        self.nlg_prompt_cls = NlgPromptFactory.get_handler(
            cfg.prompt_type, cfg.context_type
        )
        self.encoder_model = encoder_model
        self.accelerator = Accelerator()

    def trim_dialog_history(
        self,
        item: TodTurnCsvRow,
        trim_len: int,
    ):
        dialog_history_tokens = TokenizerUtilities.tokenize(
            self.tokenizer, item.context
        )
        trimmed_history_tokens = dialog_history_tokens[trim_len + 15 :]
        trimmed_history_text = self.tokenizer.decode(trimmed_history_tokens)
        generation_prompt = self.nlg_prompt_cls.get_generation_prompt(
            item.domains, trimmed_history_text
        )
        context_tokens = TokenizerUtilities.tokenize(self.tokenizer, generation_prompt)
        if len(context_tokens) > self.cfg.test_prompt_max_len:
            overflow_tokens = len(context_tokens) - self.cfg.test_prompt_max_len
            return self.trim_dialog_history(item, -overflow_tokens)
        return context_tokens

    def tod_train_collate(self, batch: list[TodTurnCsvRow]):
        all_input_tokens = []
        all_labels = []
        all_attention_masks = []
        all_schema_tokens = []

        target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
        for item in batch:
            row = self.collate_single_item(item, target_max_len)
            all_input_tokens.append(row.input_tokens)
            all_attention_masks.append(row.attention_mask)
            all_labels.append(row.label)
            all_schema_tokens.append(row.schema_tokens)
        # return DotMap(
        return {
            "input_ids": torch.stack(all_input_tokens),
            "labels": torch.stack(all_labels),
            "attention_mask": torch.stack(all_attention_masks),
            "schema_tokens": torch.stack(all_schema_tokens),
        }

    def collate_single_item(
        self,
        item: TodTurnCsvRow,
        target_max_len: int,
        is_test: bool = False,
        is_t5_model: bool = False,
    ) -> TodTrainCrossRowCollator:

        schema_prompt = self.nlg_prompt_cls.get_schema_prompt(item.domains, item.schema)
        generation_prompt = self.nlg_prompt_cls.get_generation_prompt(
            item.domains, item.context
        )
        generation_prompt_tokens = TokenizerUtilities.tokenize(
            self.tokenizer, generation_prompt
        )
        schema_tokens = TokenizerUtilities.tokenize(self.tokenizer, schema_prompt)
        unused_len = self.cfg.test_prompt_max_len - len(generation_prompt_tokens)
        if unused_len < 0:
            generation_prompt_tokens = self.trim_dialog_history(item, -unused_len)
            unused_len = self.cfg.test_prompt_max_len - len(generation_prompt_tokens)
        pad = torch.full([unused_len], self.tokenizer.pad_token_id)
        target_tokens = TokenizerUtilities.tokenize(self.tokenizer, item.target)
        target_unused_len = target_max_len - len(target_tokens)
        if target_unused_len < 0:
            raise Exception("Target is too long")
        return self.get_decoder_item(
            context_tokens=generation_prompt_tokens,
            target_tokens=target_tokens,
            schema_tokens=schema_tokens,
            pad_tokens=pad,
            target_max_len=target_max_len,
            is_test=is_test,
        )

    def get_decoder_item(
        self,
        context_tokens,
        target_tokens,
        schema_tokens,
        pad_tokens,
        target_max_len,
        is_test: bool,
    ) -> TodTrainCrossRowCollator:
        target_unused_len = target_max_len - len(target_tokens)
        target_pad = torch.full([target_unused_len], self.tokenizer.pad_token_id)
        input_items = (
            [pad_tokens, context_tokens]
            if is_test
            else [pad_tokens, context_tokens, target_tokens, target_pad]
        )
        input_tokens = torch.cat(input_items)
        attention_mask = input_tokens.ne(self.tokenizer.pad_token_id).to(torch.int32)
        if is_test:
            label = torch.cat(
                [
                    target_tokens,
                    torch.full([target_unused_len], self.tokenizer.pad_token_id),
                ]
            )
        else:
            label = torch.cat(
                [
                    torch.full(
                        [len(pad_tokens) + len(context_tokens)],
                        self._huggingface_ignore_label_id,
                    ),
                    target_tokens,
                    torch.full([target_unused_len], self._huggingface_ignore_label_id),
                ]
            )
        # encoder_hidden_states = self.get_encoder_hidden_states(schema_tokens)
        return TodTrainCrossRowCollator(
            input_tokens,
            label,
            attention_mask,
            schema_tokens=schema_tokens,
        )

    def get_encoder_hidden_states(self, schema_tokens):
        with torch.no_grad():
            encoder_outputs = self.encoder_model(
                # input_ids=self.accelerator.prepare(schema_tokens),
                input_ids=schema_tokens.cuda(),
            )
        state = encoder_outputs.last_hidden_state
        return state

    def tod_test_collate(self, batch: list[TodTurnApiCallCsvRow]):
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
                    "encoder_hidden_states",
                ]
            }
        )
        max_lengths = DotMap(
            domains=100,
        )
        target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
        for item in batch:
            row = self.collate_single_item(item, target_max_len, is_test=True)
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
            domain_tokens = self.my_tokenizer_pad(
                item.domains_original, max_lengths.domains
            )
            data.domain_ids.append(domain_tokens["input_ids"][0])
            data.encoder_hidden_states.append(row.encoder_hidden_states)

        return CrossTestDataBatch(
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
            encoder_hidden_states=torch.stack(data.encoder_hidden_states),
        )
