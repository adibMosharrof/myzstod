from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from my_enums import SpecialTokens


@dataclass_json
@dataclass
class MultiWozSlot:
    name: str
    description: str
    is_categorical: bool
    possible_values: Optional[list[str]] = None

    def __eq__(self, slot_name: str) -> bool:
        return self.name == slot_name

    def __str__(self):
        if not self.possible_values:
            return self.name
        return "".join(
            [
                self.name,
                SpecialTokens.possible_values,
                " ".join(self.possible_values),
                SpecialTokens.end_possible_values,
            ]
        )


@dataclass_json
@dataclass
class MultiWozSchema:
    service_name: str
    slots: list[MultiWozSlot]

    def __eq__(self, other: any) -> bool:
        return self.service_name == other.service_name

    def __str__(self):
        return "".join(
            [
                SpecialTokens.begin_schema,
                self.service_name,
                SpecialTokens.begin_schema_slot,
                " ".join(map(str, self.slots)),
                SpecialTokens.end_schema,
            ]
        )
