from dataclasses import dataclass, field
from typing import List, Dict, Optional
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

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return " ".join([self.domain, self.slot_name, self.value])


@dataclass
class TodAction:
    domain: str
    action_type: str
    slot_name: str

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return " ".join([self.domain, self.action_type, self.slot_name])


@dataclass
class TodContext:
    user_utterances: List[str] = field(default_factory=list)
    system_utterances: List[str] = field(default_factory=list)
    next_system_utterance: str = None

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        out = SpecialTokens.begin_context
        for user, system in zip_longest(
            self.user_utterances, self.system_utterances, fillvalue=""
        ):
            if user:
                out += SpecialTokens.user + user
            if system:
                out += SpecialTokens.system + system
        out += SpecialTokens.end_context
        return out


@dataclass
class TodTarget:
    beliefs: List[TodBelief]
    actions: List[TodAction]
    response: str

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        out = SpecialTokens.begin_belief
        out += ", ".join(map(str, self.beliefs))
        out += SpecialTokens.end_belief

        out += SpecialTokens.begin_action
        out += ", ".join(map(str, self.actions))
        out += SpecialTokens.end_action

        out += SpecialTokens.begin_response
        out += self.response
        out += SpecialTokens.end_response

        return out


@dataclass
class TodTurn:
    context: TodContext
    target: TodTarget
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None

    def to_csv_row(self) -> List[any]:
        return [self.dialog_id, self.turn_id, str(self.context), str(self.target)]

"""
    End of Simple Tod data prep classes
"""


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
