from my_enums import ContextType
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase


class ApiCallTurnCsvRow(TurnCsvRowBase):
    def get_csv_headers(self, should_add_schema: bool = True) -> list[str]:
        headers = super().get_csv_headers(should_add_schema)
        headers += ["target", "turn_row_type", "is_retrieval", "is_slot_fill"]
        return headers

    def to_csv_row(
        self, context_type: ContextType, tod_turn: NlgTodTurn, should_add_schema: bool
    ) -> list[str]:
        row = super().to_csv_row(context_type, tod_turn, should_add_schema)
        row += [int(tod_turn.turn_row_type), tod_turn.is_retrieval, tod_turn.is_slot_fill]
        return row
