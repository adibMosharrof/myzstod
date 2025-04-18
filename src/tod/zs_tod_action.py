from dataclasses import dataclass
from typing import Optional
from sgd_dstc8_data_model.dstc_dataclasses import DstcRequestedSlot

from my_enums import ZsTodConstants
import dstc.dstc_utils as dstc_utils


@dataclass
class ZsTodAction:
    domain: str
    action_type: str
    slot_name: Optional[str] = ""
    values: Optional[str] = ""
    prediction: Optional[str] = ""
    is_categorical: Optional[bool] = None

    @classmethod
    def from_string(
        self, text: str, slot_categories: dict[str, bool] = None
    ) -> "ZsTodAction":
        try:
            action_type, rest = text.split(ZsTodConstants.SLOT_VALUE_SEPARATOR)
        except ValueError:
            return self("", "", prediction=text)
        try:
            dom_slot, values = rest.split(ZsTodConstants.ACTION_VALUE_SEPARATOR)
            # values = [v.strip() for v in values]
        except ValueError:
            return self("", action_type, prediction=text)
        try:
            domain, slot_name = dom_slot.split(ZsTodConstants.DOMAIN_SLOT_SEPARATOR)
        except ValueError:
            return self("", action_type, values, text)
        is_categorical = None
        if slot_categories:
            is_categorical = slot_categories[slot_name]
        return self(
            domain.strip(),
            action_type.strip(),
            slot_name.strip(),
            values,
            is_categorical=is_categorical,
        )

    def __eq__(self, other: "ZsTodAction") -> bool:
        if isinstance(other, DstcRequestedSlot):
            return self.domain == other.domain and self.slot_name == other.slot_name
        return (
            self.domain == other.domain
            and self.action_type == other.action_type
            and self.slot_name == other.slot_name
            and dstc_utils.get_slot_value_match_score(
                self.values, other.values, self.is_categorical
            )
        )

    def is_inform(self) -> bool:
        return (
            self.action_type == ZsTodConstants.ACTION_TYPE_INFORM
            or self.action_type == ZsTodConstants.ACTION_TYPE_INFORM_COUNT
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "".join(
            [
                self.action_type,
                ZsTodConstants.SLOT_VALUE_SEPARATOR,
                self.domain,
                ZsTodConstants.DOMAIN_SLOT_SEPARATOR,
                self.slot_name,
                ZsTodConstants.ACTION_VALUE_SEPARATOR,
                self.values,
            ]
        )
