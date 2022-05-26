from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dataclasses_json import dataclass_json
from enum import Enum
from itertools import zip_longest

"""
    Simple Tod data prep classes
"""


@dataclass
class TodBelief:
    domain: str
    slot_name: str
    value: any


@dataclass
class TodAction:
    domain: str
    action_type: str
    slot_name: str


@dataclass
class TodContext:
    user_utterances: List[str] = field(default_factory=list)
    system_utterances: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        out = SpecialTokens.begin_context
        for user, system in zip_longest(
            self.user_utterances, self.system_utterances, fillvalue=""
        ):
            u = SpecialTokens.user + user
            s = SpecialTokens.system + system
            out += u + s
        out += SpecialTokens.end_context
        return out


@dataclass
class TodTarget:
    beliefs: List[TodBelief]
    actions: List[TodAction]
    response: str


@dataclass
class TodTurn:
    context: TodContext
    target: TodTarget


"""
    End of Simple Tod data prep classes
"""

"""
    DSTC Dialog Dataclass
"""


@dataclass
class DstcState:
    active_intent: str
    slot_values: Dict[str, any]
    requested_slot: Optional[List[any]] = None


@dataclass
class DstcAction:
    act: str
    canonical_values: List[str]
    slot: str
    values: List[str]
    service_call: Optional[any] = None
    service_results: Optional[any] = None


@dataclass
class DstcFrame:
    actions: List[DstcAction]
    service: str
    slots: List[any]
    state: Optional[DstcState] = None


@dataclass
class DstcTurn:
    frames: List[DstcFrame]
    speaker: str
    utterance: str


@dataclass_json
@dataclass
class DstcDialog:
    dialogue_id: str
    services: List[str]
    turns: List[DstcTurn]


class Speaker(str, Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"


class SpecialTokens(str, Enum):
    begin_context = "<|begincontext|>"
    end_context = "<|endcontext|>"
    system = "<|system|>"
    user = "<|user|>"

    begin_belief = "<|beginbelief|>"
    end_belief = "<|endbelief|>"

    begin_response = "<|beginresponse|>"
    end_response = "<|endresponse|>"

    begin_action = "<|beginaction|>"
    end_action = "<|endaction|>"
