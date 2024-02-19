from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Optional

from my_enums import TurnRowType

from sgd_dstc8_data_model.dstc_dataclasses import DstcServiceCall


@dataclass
class BiTodContext:
    dialog_history: str
    current_user_utterance: str
    service_results: Optional[list[dict[str, str]]] = field(default_factory=list)
    turn_row_type: Optional[TurnRowType] = None
    api_call: Optional[dict[str, str]] = None

    def _get_last_user_utterance(self) -> str:
        return "".join(["Last User Utterance:", self.current_user_utterance])

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
                self.api_call if self.api_call else "",
                self.get_service_results(),
            ]
        )
