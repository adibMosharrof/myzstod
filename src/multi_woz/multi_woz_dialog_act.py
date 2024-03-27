from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass
class MultiWozAct:
    action_type: str
    slot_name: str
    value: str


@dataclass
class MultiWozActTurn:
    actions: list[MultiWozAct]
    turn_id: str


@dataclass_json
@dataclass
class MultiWozDialogAct:
    dialogue_id: str
    turns: dict[str, MultiWozAct]
