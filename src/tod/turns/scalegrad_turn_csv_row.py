from typing import Union

import numpy as np
from my_enums import ContextType
from tod.turns.general_turn_csv_row import GeneralTurnCsvRow
from tod.turns.multi_task_csv_row import MultiTaskCsvRow
from tod.turns.turn_csv_row_base import TurnCsvRowBase
from tod.turns.zs_tod_turn import ZsTodTurn

""" Prepares the csv rows for scale grad turns

This class is needs to be instantiated from [GeneralTurnCsvRow, MultiTaskCsvRow]
Adds a column to csv for special words that should be given importance during scale grad boosting
"""


class ScaleGradTurnCsvRow(TurnCsvRowBase):
    def __init__(self, base_class: Union[GeneralTurnCsvRow, MultiTaskCsvRow]):
        super().__init__()
        self.base_class = base_class

    def get_csv_headers(self, should_add_schema: bool) -> list[str]:
        headers = self.base_class.get_csv_headers(should_add_schema)
        headers.append("special_tokens")
        return headers

    def get_special_words_from_schema(self, tod_turn: ZsTodTurn) -> str:
        special_words = []
        for schema in tod_turn.schemas:
            intents = schema.get_intents()
            slots = schema.get_slot_names()
            special_words.append(intents)
            special_words.append(slots)
        all_words = np.concatenate(special_words, axis=0)
        return "|".join(all_words)

    def to_csv_row(
        self, context_type: ContextType, tod_turn: ZsTodTurn, should_add_schema: bool
    ) -> list[str]:
        row = self.base_class.to_csv_row(context_type, tod_turn, should_add_schema)
        special_words = self.get_special_words_from_schema(tod_turn)
        row.append(special_words)
        return row
