from dataclasses import dataclass
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchemaSlot,
    DstcSchemaIntent,
    DstcSchema,
)
from dataclasses_json import dataclass_json
from utilities.text_utilities import remove_underscore, get_nlg_service_name


@dataclass
class PseudoSchemaSlot(DstcSchemaSlot):
    pseudo_name: str


@dataclass
class PseudoSchemaIntent(DstcSchemaIntent):
    pseudo_name: str

    def nlg_repr(self, slot_map: dict[str, str]) -> str:
        name = remove_underscore(self.name)
        # required_slots = ",".join(map(remove_underscore, self.required_slots))
        # optional_slots = ",".join(map(remove_underscore, self.optional_slots))
        # pseudo_required = ",".join(map(slot_map.get, self.required_slots))
        # pseudo_optional = ",".join(map(slot_map.get, self.optional_slots))
        required_slots = []
        optional_slots = []
        for req_slot in self.required_slots:
            required_slots.append(
                f"{remove_underscore(req_slot)} ({slot_map.get(req_slot)})"
            )
        for opt_slot in self.optional_slots:
            optional_slots.append(
                f"{remove_underscore(opt_slot)} ({slot_map.get(opt_slot)})"
            )
        req_slots_text = ", ".join(required_slots)
        opt_slots_text = ", ".join(optional_slots)
        return "\n".join(
            [
                f"Intent ({name},{self.pseudo_name})",
                f"required slots: {req_slots_text}" if required_slots else "",
                f"optional slots: {opt_slots_text}" if optional_slots else "",
            ]
        )
        return "\n".join(
            [
                f"Intent {name}",
                f"Intent pseudo name: {self.pseudo_name}",
                f"required slot names: {required_slots}" if required_slots else "",
                (
                    f"required slot pseudo names: {pseudo_required}"
                    if pseudo_required
                    else ""
                ),
                f"optional slots: {optional_slots}" if optional_slots else "",
                (
                    f"optional slot pseudo names: {pseudo_optional}"
                    if pseudo_optional
                    else ""
                ),
            ]
        )


@dataclass
@dataclass_json
class PseudoSchema(DstcSchema):
    slots: list[PseudoSchemaSlot]
    intents: list[PseudoSchemaIntent]

    def get_pseudo_intent_name(self, intent_name: str) -> str:
        for intent in self.intents:
            if intent.name == intent_name:
                return intent.pseudo_name
        return None

    def get_pseudo_slot_name(self, slot_name: str) -> str:
        for slot in self.slots:
            if slot.name == slot_name:
                return slot.pseudo_name
        return None
        raise ValueError(f"Slot {slot_name} not found in schema {self.service_name}")

    def get_nlg_repr(self) -> str:
        slot_map = {}
        for slot in self.slots:
            slot_map[slot.name] = slot.pseudo_name
        return "\n".join(
            [
                f"Schema name: {get_nlg_service_name(self.service_name)}",
                "\n".join([intent.nlg_repr(slot_map) for intent in self.intents]),
            ]
        )
