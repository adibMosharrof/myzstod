from dataclasses import dataclass, field
from enum import Enum
from itertools import zip_longest
from typing import Dict, List, Optional
from collections import deque
import torch


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
    user_utterances: deque[str] = field(default_factory=deque)
    system_utterances: deque[str] = field(default_factory=deque)
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
    active_intent: Optional[str] = None
    requested_slots: Optional[List[str]] = None

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        out = SpecialTokens.begin_target

        if self.active_intent:
            out += SpecialTokens.begin_intent
            out += self.active_intent
            out += SpecialTokens.end_intent + "\n\n"

        if self.requested_slots:
            out = SpecialTokens.begin_requested_slots
            out += ", ".join(map(str, self.requested_slots))
            out += SpecialTokens.end_requested_slots + "\n\n"

        out += SpecialTokens.begin_belief
        out += ", ".join(map(str, self.beliefs))
        out += SpecialTokens.end_belief + "\n\n"

        out += SpecialTokens.begin_action
        out += ", ".join(map(str, self.actions))
        out += SpecialTokens.end_action + "\n\n"

        out += SpecialTokens.begin_response
        out += self.response
        out += SpecialTokens.end_response + "\n\n"

        # out += SpecialTokens.end_of_text
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
    begin_target = "<|begintarget|>"

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

    begin_intent = "<|beginintent|>"
    end_intent = "<|endintent|>"

    begin_requested_slots = "<|beginrequestedslots|>"
    end_requested_slots = "<|endrequestedslots|>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class TokenizerTokens(str, Enum):
    pad_token = "<|pad|>"
    eos_token = "<|endoftext|>"
    bos_token = "<|startoftext|>"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.__str__()


class SimpleTodConstants(str, Enum):
    DELEXICALIZED = "_delexicalized"


# Datamodule classes


@dataclass
class SimpleTodDatasetItem:
    context: str
    target: str


@dataclass
class SimpleTodTestDataRow:
    context_tokens: torch.Tensor
    context_attention_masks: torch.Tensor
    label_tokens: torch.Tensor
    label_attention_masks: torch.Tensor
    contexts_text: str
    targets_text: str


@dataclass
class SimpleTodTestDataBatch:
    context_tokens: torch.Tensor
    context_attention_masks: torch.Tensor
    label_tokens: torch.Tensor
    label_attention_masks: torch.Tensor
    contexts_text: List[str]
    targets_text: List[str]

    def __iter__(self):
        for item in zip(
            self.context_tokens,
            self.context_attention_masks,
            self.label_tokens,
            self.label_attention_masks,
            self.contexts_text,
            self.targets_text,
        ):
            yield SimpleTodTestDataRow(*item)
