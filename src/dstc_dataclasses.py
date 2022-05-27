from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dataclasses_json import dataclass_json
from enum import Enum
from itertools import zip_longest

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



