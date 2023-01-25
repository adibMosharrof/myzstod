
import numpy as np
from multi_head.mh_dataclasses import MultiHeadDictFactory
from my_enums import ContextType
from tod.turns.turn_csv_row import TurnCsvRowBase
from tod.turns.zs_tod_turn import ZsTodTurn

class MhTurnCsvRow(TurnCsvRowBase):
    def __init__(self, mh_fact: MultiHeadDictFactory):
        self.mh_fact = mh_fact

    def get_csv_headers(self, should_add_schema: bool)->list[str]:
        headers= super().get_csv_headers(should_add_schema)
        return headers + self.mh_fact.get_head_names()

    def to_csv_row(self, context_type:ContextType, tod_turn: ZsTodTurn, should_add_schema: bool)->list[str]:
        row = super().to_csv_row(context_type, tod_turn, should_add_schema)
        mh_target = [
                getattr(tod_turn.target, mhi.target_attr)() for mhi in self.mh_fact.get_head_instances()
        ]
        return row + mh_target
