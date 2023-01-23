
from my_enums import ContextType
from simple_tod_dataclasses import SimpleTodTurn
from tod.turns.turn_csv_row import TurnCsvRowBase

class GeneralTurnCsvRow(TurnCsvRowBase):
    
    def get_csv_headers(self, should_add_schema: bool)->list[str]:
        headers= super().get_csv_headers(should_add_schema)
        headers.append("target")
        return headers

    def to_csv_row(self, context_type:ContextType, tod_turn: SimpleTodTurn)->list[str]:
        row = super().to_csv_row(context_type, tod_turn)
        target_str = str(tod_turn.target)
        row.append(target_str)
        return row
