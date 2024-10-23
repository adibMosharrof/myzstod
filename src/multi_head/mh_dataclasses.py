from dataclasses import dataclass, field
from enum import Enum
from dataclasses_json import dataclass_json
from torch import nn
from typing import Union, Optional
from my_enums import ZsTodConstants, SpecialTokens, MultiHeadName
from transformers import AutoTokenizer
import dstc.dstc_utils as dstc_utils


@dataclass
class MultiHeadInstance:
    name: str
    eos_token_id: Optional[int] = None
    repr: Optional[str] = None
    target_attr: Optional[str] = None
    prompt_token: Optional[str] = None
    bad_word_tokens: Optional[list[str]] = None
    ##order of this is important, as they will be concatenated in this order during training
    head_dependencies: Optional[list[str]] = field(default_factory=list)

@dataclass
class MultiHeadDictFactory:
    def __init__(self, tokenizer: AutoTokenizer):
        self.multi_head_instances = {
            MultiHeadName.DSTS.value: MultiHeadInstance(
                MultiHeadName.DSTS.value,
                dstc_utils.get_token_id(tokenizer, SpecialTokens.end_dsts.value),
                target_attr="get_dsts",
                prompt_token=SpecialTokens.begin_dsts,
                bad_word_tokens=MhBadWordTokens.dsts(),
            ),
            MultiHeadName.SYSTEM_ACTIONS.value: MultiHeadInstance(
                MultiHeadName.SYSTEM_ACTIONS.value,
                dstc_utils.get_token_id(tokenizer, SpecialTokens.end_action.value),
                target_attr="get_actions",
                prompt_token=SpecialTokens.begin_action,
                bad_word_tokens=MhBadWordTokens.system_actions(),
                head_dependencies=[MultiHeadName.DSTS.value],
            ),
            MultiHeadName.NLG.value: MultiHeadInstance(
                MultiHeadName.NLG.value,
                dstc_utils.get_token_id(tokenizer, SpecialTokens.end_response.value),
                target_attr="get_response",
                prompt_token=SpecialTokens.begin_response,
                bad_word_tokens=MhBadWordTokens.nlg(),
                head_dependencies=[
                    MultiHeadName.DSTS.value,
                    MultiHeadName.SYSTEM_ACTIONS.value,
                ],
            ),
        }

    def get_dependencies_of_head(self, head_name: str) -> list[MultiHeadInstance]:
        head_dependencies = self.multi_head_instances[head_name].head_dependencies
        return [self.multi_head_instances[name] for name in head_dependencies]

    def get_head_names(self) -> list[str]:
        return list(self.multi_head_instances.keys())

    def get_head_instances(self) -> list[MultiHeadInstance]:
        return list(self.multi_head_instances.values())

    def get_head_prompt_token_pairs(self) -> list[tuple[str, str]]:
        return [
            (mh.name, mh.prompt_token) for _, mh in self.multi_head_instances.items()
        ]


class MhBadWordTokens:
    @classmethod
    def system_actions(self):
        return self._dsts_special_tokens() + self._nlg_special_tokens()

    @classmethod
    def dsts(self):
        return self._action_special_tokens() + self._nlg_special_tokens()

    @classmethod
    def nlg(self):
        return (
            self._dsts_special_tokens()
            + self._action_special_tokens()
            + [
                ZsTodConstants.SLOT_VALUE_SEPARATOR,
                ZsTodConstants.DOMAIN_SLOT_SEPARATOR,
                ZsTodConstants.ACTION_VALUE_SEPARATOR,
                ZsTodConstants.ITEM_SEPARATOR,
            ]
        )

    @classmethod
    def _action_special_tokens(self):
        return [SpecialTokens.begin_action, SpecialTokens.end_action]

    @classmethod
    def _dsts_special_tokens(self):
        return [
            SpecialTokens.begin_dsts,
            SpecialTokens.end_dsts,
            SpecialTokens.begin_dst,
            SpecialTokens.end_dst,
            SpecialTokens.begin_belief,
            SpecialTokens.end_belief,
            SpecialTokens.begin_intent,
            SpecialTokens.end_intent,
        ]

    @classmethod
    def _nlg_special_tokens(self):
        return [SpecialTokens.begin_response, SpecialTokens.end_response]
