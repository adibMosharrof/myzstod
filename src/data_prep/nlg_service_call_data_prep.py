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
from my_enums import ContextType, ZsTodConstants
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase

from utilities.text_utilities import get_nlg_service_name
import utils
import data_prep.data_prep_utils as data_prep_utils


class NlgServiceCallDataPrep(DataPrepStrategy):
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

    def prepare_service_call_target(self):
        pass

    def prepare_service_call_context(self):
        pass

    def get_turn_schema_str(self, turn_schemas) -> str:
        return "".join([s.get_nlg_repr() for s in turn_schemas])

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
            i = self.prepare_nlg_service_call_turn(
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

    def add_tod_turn(
        self,
        turn_csv_row_handler: TurnCsvRowBase,
        tod_turns: list[NlgTodTurn],
        tod_turn: NlgTodTurn,
        dialog_id: int,
        turn_id: int,
    ):
        tod_turn.dialog_id = dialog_id
        tod_turn.turn_id = turn_id
        tod_turns.append(
            turn_csv_row_handler.to_csv_row(
                self.cfg.context_type, tod_turn, self.cfg.should_add_schema
            )
        )

    def prepare_nlg_service_call_turn(
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
        if not tod_turn.context.service_call:
            return turn_id
        new_turn = copy.deepcopy(tod_turn)
        new_turn.context.service_call = None
        new_turn.is_service_call = True
        new_turn.target.response = tod_turn.context._get_service_call()
        new_turn.dialog_id = dialog_id
        new_turn.turn_id = turn_id
        self.add_tod_turn(turn_csv_row_handler, tod_turns, new_turn, dialog_id, turn_id)
        turn_id += 1
        return turn_id
