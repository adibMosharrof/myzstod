from dataclasses import dataclass, field
from enum import Enum
from itertools import zip_longest
from typing import Dict, List, Optional


@dataclass
class SimpleTodBelief:
    domain: str
    slot_name: str
    value: any

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.domain}_{self.slot_name}: {self.value}"


@dataclass
class SimpleTodAction:
    domain: str
    action_type: str
    slot_name: str

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"{self.action_type} {self.domain}_{self.slot_name}"


@dataclass
class SimpleTodContext:
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
        out += SpecialTokens.end_context + "\n\n"
        return out


@dataclass
class SimpleTodTarget:
    beliefs: List[SimpleTodBelief]
    actions: List[SimpleTodAction]
    response: str

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        out = SpecialTokens.begin_belief
        out += ", ".join(map(str, self.beliefs))
        out += SpecialTokens.end_belief + "\n\n"

        out += SpecialTokens.begin_action
        out += ", ".join(map(str, self.actions))
        out += SpecialTokens.end_action + "\n\n"

        out += SpecialTokens.begin_response
        out += self.response
        out += SpecialTokens.end_response + "\n\n"

        return out


@dataclass
class SimpleTodTurn:
    context: SimpleTodContext
    target: SimpleTodTarget
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None

    def to_csv_row(self) -> List[any]:
        return [self.dialog_id, self.turn_id, str(self.context), str(self.target)]


@dataclass
class SimpleTodTurnCsvRow:
    dialog_id: str
    turn_id: str
    context: str
    target: str


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

    @classmethod
    def list(cls):
        return [c.value for c in cls]
        # return list(map(lambda c: c.value, cls))
