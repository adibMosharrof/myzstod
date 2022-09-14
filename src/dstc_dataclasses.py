from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dataclasses_json import dataclass_json
from enum import Enum
import humps
import dstc_utils
from my_enums import Speaker
from simple_tod_dataclasses import SimpleTodRequestedSlot


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
    slots: List[any]
    service: str
    state: Optional[DstcState] = None

    def __init__(
        self,
        actions: List[DstcAction],
        slots: List[any],
        service: str,
        state: Optional[DstcState] = None,
    ):
        self.actions = actions
        self.slots = slots
        self.state = state
        self.service = dstc_utils.get_dstc_service_name(service)


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

    def get_requested_slots(self) -> Optional[List[SimpleTodRequestedSlot]]:
        if self.speaker == Speaker.SYSTEM:
            return None
        for frame in self.frames:
            if frame.state is None or frame.state.requested_slots is None:
                continue
            if len(frame.state.requested_slots):
                return [
                    SimpleTodRequestedSlot(frame.service, humps.camelize(s))
                    for s in frame.state.requested_slots
                ]
        return None


@dataclass_json
@dataclass
class DstcDialog:
    dialogue_id: str
    turns: List[DstcTurn]
    services: List[str]

    def __init__(self, dialogue_id: str, turns: List[DstcTurn], services: List[str]):
        self.dialogue_id = dialogue_id
        self.turns = turns
        self.services = [dstc_utils.get_dstc_service_name(s) for s in services]


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
