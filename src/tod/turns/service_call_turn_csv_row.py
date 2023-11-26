from my_enums import ContextType
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase


class ServiceCallTurnCsvRow(TurnCsvRowBase):
    def get_csv_headers(self, should_add_schema: bool) -> list[str]:
        headers = super().get_csv_headers(should_add_schema)
        headers += ["target", "is_service_call"]
        return headers

    def to_csv_row(
        self, context_type: ContextType, tod_turn: NlgTodTurn, should_add_schema: bool
    ) -> list[str]:
        row = super().to_csv_row(context_type, tod_turn, should_add_schema)
        row.append(int(tod_turn.is_service_call))
        return row
