from my_enums import ContextType
from tod.nlg.ke_tod_turn import KeTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase


class KetodApiCallTurnCsvRow(TurnCsvRowBase):
    def get_csv_headers(self, should_add_schema: bool) -> list[str]:
        headers = super().get_csv_headers(should_add_schema)
        headers += [
            "target",
            "turn_row_type",
            "is_retrieval",
            "is_slot_fill",
            "is_multi_domain_api_call",
        ]
        return headers

    def to_csv_row(
        self,
        context_type: ContextType,
        tod_turn: KeTodTurn,
        should_add_schema: bool,
        step_name=None,
    ) -> list[str]:
        row = super().to_csv_row(
            context_type, tod_turn, should_add_schema, step_name=step_name
        )
        row += [
            int(tod_turn.turn_row_type),
            tod_turn.is_retrieval,
            tod_turn.is_slot_fill,
            tod_turn.is_multi_domain_api_call,
        ]
        return row
