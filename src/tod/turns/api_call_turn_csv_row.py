from dataclasses import dataclass, fields
from typing import Optional
from my_enums import ContextType
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase


@dataclass
class ApiCallTurnCsvRow(TurnCsvRowBase):
    turn_row_type: Optional[int] = None
    is_retrieval: Optional[int] = None
    is_slot_fill: Optional[int] = None
    is_multi_domain_api_call: Optional[int] = None
    dataset_name: Optional[str] = None
    is_single_domain: Optional[int] = None
    current_user_utterance: Optional[str] = None
    search_results: Optional[str] = None

    def get_csv_headers(self, should_add_schema: bool = True) -> list[str]:
        headers = super().get_csv_headers(should_add_schema)
        headers += [
            "target",
            "turn_row_type",
            "is_retrieval",
            "is_slot_fill",
            "is_multi_domain_api_call",
            "dataset_name",
            "is_single_domain",
            "current_user_utterance",
            "search_results",
        ]
        return headers

    def to_csv_row(
        self,
        context_type: ContextType,
        tod_turn: NlgTodTurn,
        should_add_schema: bool = True,
        step_name=None,
    ) -> list[str]:
        row = super().to_csv_row(
            context_type, tod_turn, should_add_schema, step_name=step_name
        )
        is_single_domain = self.get_is_single_domain(tod_turn)
        row += [
            int(tod_turn.turn_row_type),
            int(tod_turn.is_retrieval),
            tod_turn.is_slot_fill,
            tod_turn.is_multi_domain_api_call,
            tod_turn.dataset_name,
            is_single_domain,
            tod_turn.current_user_utterance,
            tod_turn.search_results,
        ]
        return row

    def get_is_single_domain(self, tod_turn: NlgTodTurn) -> bool:
        return int(len(tod_turn.domains) == 1)

    @classmethod
    def from_list_of_values_and_headers(self, values, headers):
        header_value_map = dict(zip(headers, values))
        ordered_values = [header_value_map[field.name] for field in fields(self)]
        return self(*ordered_values)
