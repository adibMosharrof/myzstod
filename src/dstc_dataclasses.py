from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dataclasses_json import dataclass_json
from enum import Enum
from itertools import zip_longest
import humps

from simple_tod_dataclasses import Speaker

"""
    DSTC Dialog Dataclass
"""


@dataclass
class DstcState:
    active_intent: str
    slot_values: Dict[str, any]
    requested_slots: Optional[List[str]] = None


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

    def get_active_intent(self) -> Optional[str]:
        if self.speaker == Speaker.SYSTEM:
            return None
        for frame in self.frames:
            if frame.state is not None:
                return frame.state.active_intent
        return None

    def get_requested_slots(self) -> Optional[List[str]]:
        if self.speaker == Speaker.SYSTEM:
            return None
        for frame in self.frames:
            if frame.state is None or frame.state.requested_slots is None:
                continue
            if len(frame.state.requested_slots):
                return [humps.camelize(s) for s in frame.state.requested_slots]
        return None


@dataclass_json
@dataclass
class DstcDialog:
    dialogue_id: str
    services: List[str]
    turns: List[DstcTurn]


@dataclass
class DstcSchemaIntent:
    name: str
    description: str
    is_transactional: bool
    required_slots: List[str]
    optional_slots: any
    result_slots: List[str]

    def __init__(
        self,
        name: str,
        description: str,
        is_transactional: bool,
        required_slots: List[str],
        optional_slots: any,
        result_slots: List[str],
    ) -> None:
        self.name = name
        self.description = description
        self.is_transactional = is_transactional
        self.required_slots = required_slots
        self.optional_slots = optional_slots
        self.result_slots = result_slots


@dataclass
class DstcSchemaSlot:
    name: str
    description: str
    is_categorical: bool
    possible_values: List[str]

    def __eq__(self, slot_name: str) -> bool:
        return self.name == slot_name


@dataclass_json
@dataclass
class DstcSchema:
    service_name: str
    description: str
    slots: List[DstcSchemaSlot]
    intents: List[DstcSchemaIntent]
    step: Optional[str] = None
