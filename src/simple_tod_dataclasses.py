from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import DefaultDict, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

import torch
from dstc_dataclasses import DstcRequestedSlot, DstcSchema

from my_enums import DstcSystemActions, SimpleTodConstants, SpecialTokens
import dstc_utils


@dataclass
class SimpleTodBelief:
    domain: str
    slot_name: str
    values: any
    prediction: Optional[str] = ""

    @classmethod
    def from_string(self, text: str):
        try:
            dom_slot, values_str = text.split(SimpleTodConstants.SLOT_VALUE_SEPARATOR)
            values = values_str.split(SimpleTodConstants.VALUE_SEPARATOR)
        except ValueError:
            return self("", "", "", text)
        try:
            domain, slot_name = dom_slot.split(SimpleTodConstants.DOMAIN_SLOT_SEPARATOR)
        except ValueError:
            return self("", "", values, text)
        return self(domain, slot_name, values)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "".join(
            [
                self.domain,
                SimpleTodConstants.DOMAIN_SLOT_SEPARATOR,
                self.slot_name,
                SimpleTodConstants.SLOT_VALUE_SEPARATOR,
                SimpleTodConstants.VALUE_SEPARATOR.join(self.values),
            ]
        )


@dataclass
class SimpleTodAction:
    domain: str
    action_type: str
    slot_name: Optional[str] = ""
    values: Optional[str] = ""
    prediction: Optional[str] = ""

    @classmethod
    def from_string(self, text: str):
        try:
            action_type, rest = text.split(SimpleTodConstants.SLOT_VALUE_SEPARATOR)
        except ValueError:
            return self("", "", prediction=text)
        try:
            dom_slot, values = rest.split(SimpleTodConstants.ACTION_VALUE_SEPARATOR)
        except ValueError:
            return self("", action_type, prediction=text)
        try:
            domain, slot_name = dom_slot.split(SimpleTodConstants.DOMAIN_SLOT_SEPARATOR)
        except ValueError:
            return self("", action_type, values, text)
        return self(domain, action_type, slot_name, values)

    def __eq__(self, other: "SimpleTodAction") -> bool:
        return (
            self.domain == other.domain
            and self.action_type == other.action_type
            and self.slot_name == other.slot_name
            and self.values == other.values
        )

    def is_inform(self) -> bool:
        return (
            self.action_type == SimpleTodConstants.ACTION_TYPE_INFORM
            or self.action_type == SimpleTodConstants.ACTION_TYPE_INFORM_COUNT
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "".join(
            [
                self.action_type,
                SimpleTodConstants.SLOT_VALUE_SEPARATOR,
                self.domain,
                SimpleTodConstants.DOMAIN_SLOT_SEPARATOR,
                self.slot_name,
                SimpleTodConstants.ACTION_VALUE_SEPARATOR,
                self.values,
            ]
        )


@dataclass
class SimpleTodContext:
    user_utterances: deque[str] = field(default_factory=deque)
    system_utterances: deque[str] = field(default_factory=deque)
    next_system_utterance: str = None
    current_user_utterance: str = None
    should_add_sys_actions: bool = None

    def __init__(self, max_length: int = 10):
        self.user_utterances = deque(maxlen=max_length)
        self.system_utterances = deque(maxlen=max_length)

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

        out += (
            SpecialTokens.begin_last_user_utterance
            + self.current_user_utterance
            + SpecialTokens.end_last_user_utterance
        )
        if self.should_add_sys_actions:
            out += "".join(
                [
                    SpecialTokens.sys_actions,
                    " ".join(DstcSystemActions.list()),
                ]
            )
        out += SpecialTokens.end_context
        return out


@dataclass
class SimpleTodDst:
    beliefs: List[SimpleTodBelief]
    active_intent: str
    requested_slots: Optional[List[DstcRequestedSlot]] = None

    def __str__(self) -> str:
        out = SpecialTokens.begin_dst
        if self.active_intent:
            out += SpecialTokens.begin_intent
            out += self.active_intent
            out += SpecialTokens.end_intent + SimpleTodConstants.NEW_LINES

        if self.requested_slots:
            out += SpecialTokens.begin_requested_slots
            out += SimpleTodConstants.ITEM_SEPARATOR.join(
                map(str, self.requested_slots)
            )
            out += SpecialTokens.end_requested_slots + SimpleTodConstants.NEW_LINES

        out += SpecialTokens.begin_belief
        out += SimpleTodConstants.ITEM_SEPARATOR.join(map(str, self.beliefs))
        out += (
            SpecialTokens.end_belief
            + SpecialTokens.end_dst
            + SimpleTodConstants.NEW_LINES
        )
        return out


@dataclass
class SimpleTodTarget:
    actions: List[SimpleTodAction]
    response: str
    dsts: List[SimpleTodDst]
    requested_slots: Optional[List[DstcRequestedSlot]] = None

    def __repr__(self) -> str:

        return self.__str__()

    def __str__(self) -> str:
        out = "".join(
            [
                SpecialTokens.begin_target,
                SpecialTokens.begin_dsts,
                "".join(map(str, self.dsts)),
                SpecialTokens.end_dsts,
                SimpleTodConstants.NEW_LINES,
            ]
        )

        out += SpecialTokens.begin_action
        out += SimpleTodConstants.ITEM_SEPARATOR.join(map(str, self.actions))
        out += SpecialTokens.end_action + SimpleTodConstants.NEW_LINES

        out += SpecialTokens.begin_response
        out += self.response
        out += SpecialTokens.end_response + SimpleTodConstants.NEW_LINES

        out += SpecialTokens.end_target
        return out


@dataclass
class SimpleTodTurnCsvRow:
    dialog_id: str
    turn_id: str
    context: str
    target: str = None
    schema: Optional[str] = None


@dataclass
class MultiTaskSpecialToken:
    start_token: SpecialTokens
    end_token: SpecialTokens
    prompt_token: SpecialTokens


@dataclass
class SimpleTodTurn:
    context: SimpleTodContext
    target: SimpleTodTarget
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None
    schemas: Optional[list[DstcSchema]] = None
    multi_task_token: Optional[MultiTaskSpecialToken] = None
    active_intent: Optional[str] = None
    schema_str: Optional[str] = None

    def to_csv_row(self) -> List[any]:
        if self.schema_str:
            return [
                self.dialog_id,
                self.turn_id,
                str(self.context),
                str(self.target),
                self.schema_str,
            ]

        return [
            self.dialog_id,
            self.turn_id,
            str(self.context),
            str(self.target),
        ]


# Datamodule classes


@dataclass
class SimpleTodTestDataBatch:
    context_tokens: list[list[int]]
    context_attention_masks: list[list[int]]
    label_tokens: list[list[int]]
    label_attention_masks: list[list[int]]
    contexts_text: list[str]
    targets_text: list[str]
    dialog_ids: list[int]
    turn_ids: list[int]


@dataclass
class PredRef:
    pred: str
    ref: str


@dataclass(frozen=True, eq=True)
class MultiTaskTurnKey:
    dialog_id: str
    turn_id: int


class InferenceRecords:
    def __init__(self):
        self.preds = []
        self.dialog_ids = []
        self.turn_ids = []
        self.refs = []
        self.contexts = []
        self.is_data_concatenated = False

    def add(self, preds, refs, dialog_ids, turn_ids, contexts):
        self.preds.append(preds)
        self.dialog_ids.append(dialog_ids)
        self.turn_ids.append(turn_ids)
        self.refs.append(refs)
        self.contexts.append(contexts)

    def concat_data(self):
        self.preds = np.concatenate(self.preds, axis=0)
        self.refs = np.concatenate(self.refs, axis=0)
        self.dialog_ids = np.concatenate(self.dialog_ids, axis=0)
        self.turn_ids = np.concatenate(self.turn_ids, axis=0)
        self.contexts = np.concatenate(self.contexts, axis=0)
        self.is_data_concatenated = True

    def get_data_by_turns(self) -> Dict[MultiTaskTurnKey, list[PredRef]]:
        if not self.is_data_concatenated:
            self.concat_data()
        turns: DefaultDict[MultiTaskTurnKey, list[PredRef]] = defaultdict(list)
        for pred, ref, dialog_id, turn_id in zip(
            self.preds, self.refs, self.dialog_ids, self.turn_ids
        ):
            turns[MultiTaskTurnKey(dialog_id, turn_id)].append(PredRef(pred, ref))
        return turns

    def get_data_for_multitask(self) -> Tuple[list[str], list[str]]:
        turns = self.get_data_by_turns()
        preds: list[str] = []
        refs: list[str] = []
        for key, turn in turns.items():
            mt_preds = "".join(
                [
                    SpecialTokens.begin_target,
                    "".join([self.extract_target(t.pred) for t in turn]),
                    SpecialTokens.end_target,
                ]
            )
            mt_refs = "".join(
                [
                    SpecialTokens.begin_target,
                    "".join([self.extract_target(t.ref) for t in turn]),
                    SpecialTokens.end_target,
                ]
            )
            preds.append(mt_preds)
            refs.append(mt_refs)
        return preds, refs

    def extract_target(self, text: str) -> str:
        return dstc_utils.remove_tokens_from_text(
            text,
            [
                SpecialTokens.begin_target,
                SpecialTokens.end_target,
                SpecialTokens.bos_token,
                SpecialTokens.eos_token,
            ],
        )


def get_multi_task_special_tokens() -> list[MultiTaskSpecialToken]:
    return [
        MultiTaskSpecialToken(
            SpecialTokens.begin_intent,
            SpecialTokens.end_intent,
            SpecialTokens.prompt_intent,
        ),
        MultiTaskSpecialToken(
            SpecialTokens.begin_requested_slots,
            SpecialTokens.end_requested_slots,
            SpecialTokens.prompt_requested_slots,
        ),
        MultiTaskSpecialToken(
            SpecialTokens.begin_belief,
            SpecialTokens.end_belief,
            SpecialTokens.prompt_belief,
        ),
        MultiTaskSpecialToken(
            SpecialTokens.begin_action,
            SpecialTokens.end_action,
            SpecialTokens.prompt_action,
        ),
        MultiTaskSpecialToken(
            SpecialTokens.begin_response,
            SpecialTokens.end_response,
            SpecialTokens.prompt_response,
        ),
    ]
