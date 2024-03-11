from typing import Optional

import numpy as np
from configs.dataprep_config import DataPrepConfig
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcDialog,
    DstcFrame,
    DstcSchema,
    DstcTurn,
    get_schemas,
)
import copy
from data_prep.data_prep_strategy import DataPrepStrategy
from my_enums import ContextType, TurnRowType, ZsTodConstants
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase

from utilities.text_utilities import get_nlg_service_name
import utils
import data_prep.data_prep_utils as data_prep_utils


class NlgApiCallDataPrep(DataPrepStrategy):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg, tod_turn_cls=NlgTodTurn, tod_context_cls=NlgTodContext)
        # self.cfg = cfg

    def prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, DstcSchema],
    ) -> NlgTodTarget:
        response = self._prepare_response(system_turn)
        return NlgTodTarget(response=response)

    def prepare_api_call_target(self):
        pass

    def prepare_api_call_context(self):
        pass

    def prepare_dialog(
        self,
        dstc_dialog: DstcDialog,
        schemas: dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> Optional[list[NlgTodTurn]]:
        tod_turns = []
        tod_turn = None
        if not data_prep_utils.is_dialogue_in_domain(
            dstc_dialog.services, self.cfg.domains
        ):
            return None
        i = 1
        for user_turn, system_turn in utils.grouper(dstc_dialog.turns, 2):
            tod_turn = self.prepare_turn(
                user_turn, system_turn, tod_turn, schemas, dstc_dialog.services
            )
            if tod_turn.target.response == "":
                continue
            i = self.prepare_nlg_api_call_turn(
                turn_csv_row_handler,
                system_turn,
                tod_turn,
                tod_turns,
                dstc_dialog.dialogue_id,
                i,
            )
            self.add_tod_turn(
                turn_csv_row_handler, tod_turns, tod_turn, dstc_dialog.dialogue_id, i
            )
            i += 1
        return tod_turns

    def prepare_nlg_api_call_turn(
        self,
        turn_csv_row_handler: TurnCsvRowBase,
        system_turn: DstcTurn,
        tod_turn: NlgTodTurn,
        tod_turns: list[NlgTodTurn],
        dialog_id: int,
        turn_id: int,
    ) -> int:
        if not system_turn:
            return turn_id
        if not tod_turn.context.api_call:
            return turn_id
        new_turn = copy.deepcopy(tod_turn)
        new_turn.context.api_call = None
        new_turn.context.service_results = None
        new_turn.turn_row_type = TurnRowType.API_CALL.value
        new_turn.target.response = tod_turn.context.get_api_call()
        new_turn.dialog_id = dialog_id
        new_turn.turn_id = turn_id
        self.add_tod_turn(turn_csv_row_handler, tod_turns, new_turn, dialog_id, turn_id)
        turn_id += 1
        return turn_id
