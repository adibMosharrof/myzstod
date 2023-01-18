from dataclasses import dataclass, fields
from dataclasses_json import dataclass_json
from torch import nn
from typing import Union, Optional
from my_enums import SimpleTodConstants, SpecialTokens
from transformers import AutoTokenizer
import dstc_utils


@dataclass
class MultiHeadInstance:
    name: str
    eos_token_id: Optional[int] = None
    repr: Optional[str] = None
    target_attr: Optional[str] = None
    prompt_token: Optional[str] = None


@dataclass
class MultiHeadDictFactory:
    multi_head_instances: list[MultiHeadInstance] = None

    def __init__(self, tokenizer: AutoTokenizer):
        self.multi_head_instances = [
            MultiHeadInstance(
                "dsts",
                dstc_utils.get_token_id(tokenizer, SpecialTokens.end_dsts.value),
                target_attr="get_dsts",
                prompt_token=SpecialTokens.begin_dsts,
            ),
            MultiHeadInstance(
                "system_actions",
                dstc_utils.get_token_id(tokenizer, SpecialTokens.end_action.value),
                target_attr="get_actions",
                prompt_token=SpecialTokens.begin_action,
            ),
            MultiHeadInstance(
                "nlg",
                dstc_utils.get_token_id(tokenizer, SpecialTokens.end_response.value),
                target_attr="get_response",
                prompt_token=SpecialTokens.begin_response,
            ),
        ]

    def get_head_names(self) -> list[str]:
        return [mh.name for mh in self.multi_head_instances]

    def get_head_instances(self) -> list[MultiHeadInstance]:
        return self.multi_head_instances

    def get_head_prompt_token_pairs(self) -> list[tuple[str, str]]:
        return [(mh.name, mh.prompt_token) for mh in self.multi_head_instances]
