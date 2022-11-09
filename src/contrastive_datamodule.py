import numpy as np
from sentence_transformers import InputExample
import torch
from base_datamodule import BaseDataModule
from contrastive_dataclasses import ContrastiveTokens

from hydra_configs import DataModuleConfig
from my_enums import (
    ContrastiveConstants,
    DstcSystemActions,
    SimpleTodActionAttributes,
    SimpleTodConstants,
    SpecialTokens,
    Steps,
)
from simple_tod_dataclasses import SimpleTodAction
import dstc_utils
import random


class ContrastiveDataModule(BaseDataModule):
    steps = Steps.list()
    _huggingface_ignore_label_id = -100
    contrastive_tokenizer = None

    def __init__(
        self,
        cfg: DataModuleConfig,
    ):
        super().__init__(cfg)

    def get_contrastive_data(
        self, contrastive_tokens: ContrastiveTokens, step=Steps.TRAIN
    ) -> list[InputExample]:
        contrastive_data = []
        self.schemas = dstc_utils.get_schemas(
            self.cfg.project_root / self.cfg.raw_data_root, step.value
        )
        swap_neg_num = self.cfg.single_action_neg_samples if step == Steps.TRAIN else 1
        for item in self.cfg.datasets[step]:
            b_txt = dstc_utils.get_text_in_between(
                item.target,
                contrastive_tokens.b_start_token,
                contrastive_tokens.b_end_token,
                default_value="",
                multiple_values=contrastive_tokens.b_multiple_values,
            )
            a_txt = SimpleTodConstants.ITEM_SEPARATOR.join(
                dstc_utils.get_text_in_between(
                    item.target,
                    contrastive_tokens.a_start_token,
                    contrastive_tokens.a_end_token,
                    default_value="",
                    multiple_values=contrastive_tokens.a_multiple_values,
                )
            )
            if self.cfg.should_add_dsts:
                dsts_txt = "".join(
                    dstc_utils.get_text_in_between(
                        item.target,
                        SpecialTokens.begin_dst,
                        SpecialTokens.end_dst,
                        default_value="",
                        multiple_values=True,
                    )
                )
                a_txt += dsts_txt

            contrastive_data.append(InputExample(texts=[b_txt, a_txt], label=1.0))
            act_splits = (
                b_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
                if contrastive_tokens.contrast_with == ContrastiveConstants.USER_ACT
                else a_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
            )
            self.get_contrastive_negative_examples(
                contrastive_data, b_txt, act_splits, swap_neg_num
            )
        if len(contrastive_data) == 0:
            raise ValueError("No contrastive data found")
        return contrastive_data

    def get_contrastive_negative_examples(
        self,
        contrastive_data: list[InputExample],
        sys_act_txt: str,
        act_splits: list[str],
        swap_neg_num: int = 5,
    ):
        if len(act_splits) > 1:
            self.get_contrastive_incomplete_negative(
                contrastive_data, sys_act_txt, act_splits
            )
            self.get_contrastive_negative_swap_data(
                contrastive_data, sys_act_txt, act_splits
            )
        else:
            for _ in range(swap_neg_num):
                self.get_contrastive_negative_swap_data(
                    contrastive_data, sys_act_txt, act_splits
                )

    def get_contrastive_incomplete_negative(
        self,
        contrastive_data: list[InputExample],
        sys_act_txt: str,
        act_splits: list[str],
    ):
        num_actions = random.randint(1, len(act_splits) - 1)
        wrong_sys_acts = random.choices(act_splits, k=num_actions)
        label = num_actions / len(act_splits)
        wrong_act_texts = SimpleTodConstants.ITEM_SEPARATOR.join(wrong_sys_acts)
        contrastive_data.append(
            InputExample(texts=[wrong_act_texts, sys_act_txt], label=label)
            # InputExample(texts=[wrong_act_texts, sys_act_txt], label=0.0)
        )

    def get_contrastive_negative_swap_data(
        self,
        contrastive_data: list[InputExample],
        sys_act_txt: str,
        act_splits: list[str],
    ):
        action = SimpleTodAction.from_string(random.choice(act_splits))
        attr = random.choice(SimpleTodActionAttributes.list())
        if attr == SimpleTodActionAttributes.domain:
            action.domain = dstc_utils.get_dstc_service_name(
                random.choice([*self.schemas])
            )
        elif attr == SimpleTodActionAttributes.action_type:
            action.action_type = random.choice(
                [x for x in DstcSystemActions.list() if x != action.action_type]
            )
        elif attr == SimpleTodActionAttributes.slot_name:
            schema = self.schemas[random.choice([*self.schemas])]
            action.slot_name = random.choice(
                [s for s in schema.slots if s.name != action.slot_name]
            ).name

        contrastive_data.append(
            InputExample(texts=[str(action), sys_act_txt], label=0.0)
        )

    def contrastive_tokenize(self, text: str):
        return self.contrastive_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.cfg.contrastive_max_token_len,
        )
