import torch
from base_datamodule import BaseDataModule
from hydra_configs import DataModuleConfig
from multi_head.mh_dataclasses import MultiHeadDict
from simple_tod_dataclasses import TodTurnMultiHeadCsvRow


class MultiLMHeadDatamodule(BaseDataModule):
    def __init__(
        self,
        cfg: DataModuleConfig,
        tod_turn_row_cls=TodTurnMultiHeadCsvRow,
    ):
        super().__init__(cfg, tod_turn_row_cls)

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
        self, batch: list[TodTurnMultiHeadCsvRow], is_pretrain: bool = False
    ):
        lengths = MultiHeadDict.from_dict(
            {
                "intents": 11,
                "beliefs": 188,
                "requested_slots": 21,
                "system_actions": 147,
                "user_actions": 85,
                "nlg": 105,
            }
        )
        context_max_length = 700
        all_head_names = MultiHeadDict.head_names()
        mh_tokens = dict.fromkeys(all_head_names, [])
        input_tokens = dict.fromkeys(all_head_names, [])
        attention_masks = dict.fromkeys(all_head_names, [])
        labels = dict.fromkeys(all_head_names, [])
        for item in batch:
            for head_name in all_head_names:
                unused_len = (
                    self.cfg.max_token_len - lengths[head_name] - context_max_length
                )
                pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
                head_target_tokens, head_target_masks = self.mh_tokenizer(
                    item[head_name], max_length=lengths[head_name]
                )
                mh_tokens[head_name].append(head_target_tokens)
                context_tokens, context_masks = self.mh_tokenizer(
                    "".join([item.context, item.schema]), max_length=context_max_length
                )
                current_input_token = torch.cat(
                    [context_tokens, head_target_tokens, pad]
                )
                input_tokens[head_name].append(current_input_token)
                attention_masks[head_name].append(
                    torch.cat(
                        [context_masks, head_target_masks, torch.full([unused_len], 0)]
                    )
                )
                if is_pretrain:
                    label = current_input_token
                else:
                    label = torch.cat(
                        [
                            torch.full(
                                [context_max_length], self._huggingface_ignore_label_id
                            ),
                            head_target_tokens,
                            torch.full([unused_len], self._huggingface_ignore_label_id),
                        ]
                    )

                labels[head_name].append(label)

        # out = {
        #     head_name: {
        #         "input_ids": torch.stack(input_tokens[head_name]),
        #         "attention_mask": torch.stack(attention_masks[head_name]),
        #         "labels": torch.stack(labels[head_name]),
        #     }
        #     for head_name in all_head_names
        # }
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
