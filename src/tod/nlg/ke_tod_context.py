from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Optional

from my_enums import TurnRowType

from sgd_dstc8_data_model.dstc_dataclasses import DstcServiceCall


@dataclass
class KeTodContext:
    dialog_history: str
    current_user_utterance: str
    service_results: Optional[list[dict[str, str]]] = None
    turn_row_type: Optional[TurnRowType] = None
    entity_query: Optional[str] = None
    kg_snippets_text: Optional[list[str]] = None
    api_call: Optional[dict[str, str]] = None

    def _get_last_user_utterance(self) -> str:
        return "".join(["Last User Utterance:", self.current_user_utterance])

    def get_entity_query(self) -> DstcServiceCall:
        out = ""
        if not self.entity_query or len(self.entity_query) < 1:
            return out
        method = ""
        params = {}

        for query in self.entity_query:
            method, pname, pval = query[0].split(" : ")
            params[pname] = pval
            # queries.append(query[0])
        entity_query = DstcServiceCall(method, params)
        entity_query.__class__.__qualname__ = "EntityQuery"
        return str(entity_query)
        # return "\n".join(["Entity Queries:", "|".join(queries), "End Entity Queries"])

    def get_kg_snippets_text(self) -> str:
        out = ""
        if not self.kg_snippets_text or len(self.kg_snippets_text) < 1:
            return out
        return "\n".join(
            ["KG Snippets:", "|".join(self.kg_snippets_text), "End KG Snippets"]
        )

    def get_api_call(self) -> str:
        out = ""
        if not self.api_call:
            return out
        dstc_api_call = DstcServiceCall(
            self.api_call.method, dict(self.api_call.parameters)
        )
        dstc_api_call.__class__.__qualname__ = "ApiCall"
        return str(dstc_api_call)

    def get_service_results(self) -> str:
        out = ""
        if not self.service_results:
            return out
        for service_result in self.service_results[:1]:
            s_res = {"search_results": dict(service_result)}
            out += str(s_res)
        return "\n".join(["\nSearch Results:", out, "End Search Results"])

    def __str__(self):
        return "\n".join(
            [
                self.dialog_history,
                self._get_last_user_utterance(),
                "End Dialog History",
                self.get_api_call(),
                self.get_service_results(),
                self.get_entity_query(),
                self.get_kg_snippets_text(),
            ]
        )
