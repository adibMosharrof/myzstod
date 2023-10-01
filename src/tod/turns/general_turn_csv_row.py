from my_enums import ContextType
from tod.turns.turn_csv_row import TurnCsvRowBase
from tod.turns.zs_tod_turn import ZsTodTurn


class GeneralTurnCsvRow(TurnCsvRowBase):
    def get_csv_headers(self, should_add_schema: bool) -> list[str]:
        headers = super().get_csv_headers(should_add_schema)
        headers.append("target")
        return headers
