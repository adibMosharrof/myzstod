import torch
from base_datamodule import BaseDataModule
from configs.dm_config import DataModuleConfig
from multi_head.mh_dataclasses import MultiHeadDictFactory
from simple_tod_dataclasses import TodTestDataBatch
from tod.turns.zs_tod_turn import TodTurnMultiHeadCsvRow
from my_enums import Steps
class MultiLMHeadDatamodule(BaseDataModule):
    def __init__(
        self,
        cfg: DataModuleConfig,
        steps: list[Steps],
        mh_fact: MultiHeadDictFactory,
        tod_turn_row_cls=TodTurnMultiHeadCsvRow,
    ):
        super().__init__(cfg, tod_turn_row_cls)
        self.mh_fact = mh_fact

    def mh_tokenizer(self, item, max_length: int):
        try:
            tokens = self.cfg.tokenizer(
                item,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
            )
        except TypeError as e:
            raise ("Contrastive tokenizer failed")
        return tokens["input_ids"][0], tokens["attention_mask"][0]

    def training_collator(
        self,
        batch: list[TodTurnMultiHeadCsvRow],
        is_pretrain: bool = False,
        max_token_len=None,
    ):
        if max_token_len is None:
            max_token_len = self.cfg.max_token_len
        all_head_names = self.mh_fact.get_head_names()
        mh_tokens = {key: [] for key in all_head_names}
        input_tokens = {key: [] for key in all_head_names}
        attention_masks = {key: [] for key in all_head_names}
        labels = {key: [] for key in all_head_names}
        target_txt = {key: [] for key in all_head_names}
        for item in batch:
            for head_name in all_head_names:
                head_target_tokens = self.train_tokenizer(item[head_name])[0]
                head_dependencies = self.mh_fact.get_dependencies_of_head(head_name)
                head_dep_texts = "".join([item[dep.name] for dep in head_dependencies])
                context_tokens = self.train_tokenizer(
                    "".join([item.schema, item.context, head_dep_texts])
                )[0]
                target_len = len(head_target_tokens)
                context_len = len(context_tokens)
                unused_len = max_token_len - context_len - target_len
                if unused_len < 0:
                    ValueError("unused length < 0")
                pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
                mh_tokens[head_name].append(head_target_tokens)
                current_input_token = torch.cat(
                    [context_tokens, head_target_tokens, pad]
                )
                input_tokens[head_name].append(current_input_token)
                attention_masks[head_name].append(
                    torch.cat(
                        [
                            torch.full([context_len + target_len], 1),
                            torch.full([unused_len], 0),
                        ]
                    )
                )
                if is_pretrain:
                    label = current_input_token
                else:
                    label = torch.cat(
                        [
                            torch.full(
                                [context_len], self._huggingface_ignore_label_id
                            ),
                            head_target_tokens,
                            torch.full([unused_len], self._huggingface_ignore_label_id),
                        ]
                    )
                target_txt[head_name].append(item[head_name])
                labels[head_name].append(label)
        out = {
            "input_ids": {
                head_name: torch.stack(input_tokens[head_name])
                for head_name in all_head_names
            },
            "attention_mask": {
                head_name: torch.stack(attention_masks[head_name])
                for head_name in all_head_names
            },
            "labels": {
                head_name: torch.stack(labels[head_name])
                for head_name in all_head_names
            },
        }

        return out

    def my_test_collate(
        self,
        batch: list[TodTurnMultiHeadCsvRow],
        max_token_len=None,
    ):
        all_head_names = self.mh_fact.get_head_names()
        input_tokens = {key: [] for key in all_head_names}
        attention_masks = {key: [] for key in all_head_names}
        all_target_txts = []
        all_context_txts = []
        turn_ids = []
        dialog_ids = []
        all_schema_txts = []
        for item in batch:
            item_target_txts = []
            for head_name, prompt_token in self.mh_fact.get_head_prompt_token_pairs():
                context_tokens = self.train_tokenizer(
                    "".join([item.schema, item.context, prompt_token])
                    # "".join([item.context, item.schema])
                )[0]
                context_len = len(context_tokens)
                unused_len = self.cfg.test_prompt_max_len - context_len
                if unused_len < 0:
                    ValueError("unused length < 0")
                pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
                current_input_token = torch.cat([context_tokens, pad])
                input_tokens[head_name].append(current_input_token)
                attention_masks[head_name].append(
                    torch.cat(
                        [
                            torch.full([context_len], 1),
                            torch.full([unused_len], 0),
                        ]
                    )
                )
                item_target_txts.append(item[head_name])

            all_target_txts.append("".join(item_target_txts))
            turn_ids.append(item.turn_id)
            all_context_txts.append(item.context)
            dialog_ids.append(item.dialog_id)
            all_schema_txts.append(item.schema)
        return TodTestDataBatch(
            input_ids={
                head_name: torch.stack(input_tokens[head_name])
                for head_name in all_head_names
            },
            attention_masks={
                head_name: torch.stack(attention_masks[head_name])
                for head_name in all_head_names
            },
            targets_text=all_target_txts,
            turn_ids=turn_ids,
            contexts_text=all_context_txts,
            dialog_ids=dialog_ids,
            schemas_text=all_schema_txts,
        )
