import numpy as np
from sentence_transformers import InputExample
import torch
from base_datamodule import BaseDataModule
from contrastive.contrastive_utils import ContrastiveTokens

from hydra_configs import DataModuleConfig
from my_datamodules import TodDataModule
from my_enums import (
    ContrastiveConstants,
    DstcSystemActions,
    ZsTodActionAttributes,
    SimpleTodConstants,
    SpecialTokens,
    Steps,
)
import dstc_utils
import random
from dstc_dataclasses import get_schemas
from itertools import combinations
from tod.turns.zs_tod_turn import TodTurnCsvRow
from tod.zs_tod_action import ZsTodAction

class ContrastiveDataModule(TodDataModule):
    steps = Steps.list()
    _huggingface_ignore_label_id = -100
    contrastive_tokenizer = None

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
        context_token_ids = []
        context_att_masks = []
        mt_prompt_ids = []
        context_max_len = self.cfg.contrastive_max_token_len
        for item in batch:
            context_tokens = self.train_tokenizer(item.context)[0]
            target_tokens = self.train_tokenizer(item.target)[0]
            schema_tokens = self.train_tokenizer(item.schema)[0]
            context_len = len(context_tokens)
            target_len = len(target_tokens)
            schema_len = len(schema_tokens)
            unused_len = self.cfg.max_token_len - context_len - target_len - schema_len

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
            context_token_id = torch.cat(
                [
                    context_tokens,
                    torch.full(
                        [context_max_len - len(context_tokens)],
                        self.cfg.tokenizer.pad_token_id,
                    ),
                ]
            )
            context_att_mask = torch.cat(
                [
                    torch.full([len(context_tokens)], 1),
                    torch.full([context_max_len - len(context_tokens)], 0),
                ]
            )

            input_ids.append(input_tokens)
            attention_masks.append(attention_mask)
            labels.append(label)
            targets_text.append(item.target)
            context_token_ids.append(context_token_id)
            context_att_masks.append(context_att_mask)

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
            "context_ids": torch.stack(context_token_ids),
            "context_attention_mask": torch.stack(context_att_masks),
        }

        if not is_pretrain and self.cfg.contrast_with and self.cfg.is_multi_task:
            out["mt_prompt_token_ids"] = torch.tensor(mt_prompt_ids)

        return out

    def get_contrastive_data(
        self, contrastive_tokens: ContrastiveTokens, step=Steps.TRAIN
    ) -> list[InputExample]:
        contrastive_data = []
        self.schemas = get_schemas(
            self.cfg.project_root / self.cfg.raw_data_root, step.value
        )
        swap_neg_num = self.cfg.single_action_neg_samples if step == Steps.TRAIN else 1
        for item in self.cfg.datasets[step]:
            if self.cfg.is_multi_task:
                if SpecialTokens.prompt_action not in item.context:
                    continue
            b_txt = dstc_utils.get_text_in_between(
                item.target,
                contrastive_tokens.b_start_token,
                contrastive_tokens.b_end_token,
                default_value="",
                multiple_values=contrastive_tokens.b_multiple_values,
            )
            source_txt = (
                item.context
                if contrastive_tokens.a_start_token
                == SpecialTokens.begin_last_user_utterance
                else item.target
            )
            a_txt = dstc_utils.get_text_in_between(
                source_txt,
                contrastive_tokens.a_start_token,
                contrastive_tokens.a_end_token,
                default_value="",
                multiple_values=contrastive_tokens.a_multiple_values,
            )
            a_dst_txt = a_txt
            if self.cfg.should_add_dsts:
                dsts_txt = "".join(
                    dstc_utils.get_text_in_between(
                        item.context,
                        SpecialTokens.begin_dst,
                        SpecialTokens.end_dst,
                        default_value="",
                        multiple_values=True,
                    )
                )
                a_dst_txt += dsts_txt

            contrastive_data.append(InputExample(texts=[a_dst_txt, b_txt], label=1.0))
            self.get_contrastive_negative_examples(
                contrastive_data, a_txt, b_txt, contrastive_tokens, swap_neg_num
            )
        if len(contrastive_data) == 0:
            raise ValueError("No contrastive data found")
        return contrastive_data

    def get_contrastive_negative_examples(
        self,
        contrastive_data: list[InputExample],
        a_txt: str,
        b_txt: list[str],
        contrastive_tokens: ContrastiveTokens,
        swap_neg_num: int = 5,
    ):
        act_txt, other_txt = self._get_act_and_other(a_txt, b_txt, contrastive_tokens)
        act_splits = act_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
        # if len(act_splits) > 1:
        #     self.get_contrastive_incomplete_negative_old(
        #         contrastive_data, other_txt, act_splits
        #     )
        #     self.get_contrastive_negative_swap_data(
        #         contrastive_data, other_txt, act_splits
        #     )
        # else:
        for _ in range(swap_neg_num):
            self.get_contrastive_negative_swap_data(
                contrastive_data, other_txt, act_splits
            )

    def _get_act_and_other(
        self, a_txt: str, b_txt: str, contrastive_tokens: ContrastiveTokens
    ) -> tuple[str, str]:
        """
        If contrast with user act,
            randomly select UA/SA for action splitting and creating negative examples
            bool random get true, split SA, false split UA
        If contrast with system act, always split on SA
        """
        if contrastive_tokens.contrast_with == ContrastiveConstants.USER_ACT:
            if bool(random.getrandbits(1)):
                return b_txt, a_txt
        return a_txt, b_txt
        # else:
        #     return a_txt, b_txt

    def get_contrastive_incomplete_negative_old(
        self,
        contrastive_data: list[InputExample],
        sys_act_txt: str,
        act_splits: list[str],
    ):
        # used_acts = set()

        num_actions = random.randint(1, len(act_splits) - 1)
        wrong_sys_acts = random.choices(act_splits, k=num_actions)
        # wrong_sys_acts_tuple = tuple(wrong_sys_acts)
        # if wrong_sys_acts_tuple in used_acts:
        #     continue
        # used_acts.add(wrong_sys_acts_tuple)
        label = num_actions / len(act_splits)
        wrong_act_texts = SimpleTodConstants.ITEM_SEPARATOR.join(wrong_sys_acts)
        contrastive_data.append(
            InputExample(texts=[wrong_act_texts, sys_act_txt], label=label)
        )

    def get_contrastive_incomplete_negative(
        self,
        contrastive_data: list[InputExample],
        sys_act_txt: str,
        act_splits: list[str],
    ):
        for i in range(1, len(act_splits) - 1):
            comb = list(combinations(act_splits, i))
            for wrong_sys_acts in comb:
                label = len(wrong_sys_acts) / len(act_splits)
                wrong_act_texts = SimpleTodConstants.ITEM_SEPARATOR.join(wrong_sys_acts)
                contrastive_data.append(
                    InputExample(texts=[wrong_act_texts, sys_act_txt], label=label)
                )

    def get_contrastive_negative_swap_data_old(
        self,
        contrastive_data: list[InputExample],
        sys_act_txt: str,
        act_splits: list[str],
    ):
        action = ZsTodAction.from_string(random.choice(act_splits))
        attr = random.choice(ZsTodActionAttributes.list())
        if attr == ZsTodActionAttributes.domain:
            action.domain = dstc_utils.get_dstc_service_name(
                random.choice([*self.schemas])
            )
        elif attr == ZsTodActionAttributes.action_type:
            action.action_type = random.choice(
                [x for x in DstcSystemActions.list() if x != action.action_type]
            )
        elif attr == ZsTodActionAttributes.slot_name:
            schema = self.schemas[random.choice([*self.schemas])]
            action.slot_name = random.choice(
                [s for s in schema.slots if s.name != action.slot_name]
            ).name

        contrastive_data.append(
            InputExample(texts=[str(action), sys_act_txt], label=0.0)
        )

    def get_contrastive_negative_swap_data(
        self,
        contrastive_data: list[InputExample],
        sys_act_txt: str,
        act_splits: list[str],
    ):
        for i, action_txt in enumerate(act_splits):
            action = ZsTodAction.from_string(action_txt)
            attr = random.choice(ZsTodActionAttributes.list())
            if attr == ZsTodActionAttributes.domain:
                action.domain = dstc_utils.get_dstc_service_name(
                    random.choice([*self.schemas])
                )
            elif attr == ZsTodActionAttributes.action_type:
                action.action_type = random.choice(
                    [x for x in DstcSystemActions.list() if x != action.action_type]
                )
            elif attr == ZsTodActionAttributes.slot_name:
                schema = self.schemas[random.choice([*self.schemas])]
                action.slot_name = random.choice(
                    [s for s in schema.slots if s.name != action.slot_name]
                ).name
            actions = act_splits.copy()
            actions[i] = str(action)
            contrastive_data.append(
                InputExample(texts=[str(actions), sys_act_txt], label=0.0)
            )

    def contrastive_tokenize(self, text: str):
        return self.contrastive_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.cfg.contrastive_max_token_len,
        )
