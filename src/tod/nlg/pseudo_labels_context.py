from dataclasses import dataclass
from tod.nlg.nlg_tod_context import NlgTodContext
from sgd_dstc8_data_model.dstc_dataclasses import DstcServiceCall


@dataclass
class PseudoLabelsContext(NlgTodContext):
    def __init__(self, max_length: int = 10):
        super().__init__(max_length)

    def get_api_call(self, schemas, turn_domains) -> str:
        out = ""
        if not self.api_call:
            return out
        turn_schemas = [schemas[domain] for domain in turn_domains]
        active_schema = None
        for schema in turn_schemas:
            pseudo_intent_name = schema.get_pseudo_intent_name(self.api_call.method)
            if pseudo_intent_name:
                active_schema = schema
                break
        pseudo_parameters = {}
        for slot_name, slot_value in self.api_call.parameters.items():
            pseudo_slot_name = active_schema.get_pseudo_slot_name(slot_name)
            pseudo_parameters[pseudo_slot_name] = slot_value

        dstc_api_call = DstcServiceCall(pseudo_intent_name, pseudo_parameters)
        dstc_api_call.__class__.__qualname__ = "ApiCall"
        return str(dstc_api_call)
