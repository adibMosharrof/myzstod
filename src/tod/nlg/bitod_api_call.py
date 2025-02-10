from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class BitodApiCallParams:
    slot_name: str
    relation: str
    value: Union[str, list[str]]

    def __post_init__(self):
        if type(self.value) == list:
            self.value = " ".join(self.value).strip()

    def __str__(self):
        return f"{self.slot_name} {self.relation} {self.value}"

    def get_by_slot_name(
        self, items: list["BitodApiCallParams"]
    ) -> Optional["BitodApiCallParams"]:
        for item in items:
            if item.slot_name == self.slot_name:
                return item
        return None


@dataclass
class BitodApiCall:
    method: str
    parameters: list[BitodApiCallParams]

    def __str__(self):
        params_joined = "|".join([str(param) for param in self.parameters])
        return f"ApiCall(method={self.method}, parameters={{{params_joined}}})"
