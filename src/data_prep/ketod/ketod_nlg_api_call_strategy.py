from configs.dataprep_config import DataPrepConfig
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.nlg_api_call_data_prep import NlgApiCallDataPrep
from my_enums import TurnRowType
from tod.nlg.ke_tod_context import KeTodContext
from tod.nlg.ke_tod_turn import KeTodTurn
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_turn import NlgTodTurn
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcDialog,
    DstcFrame,
    DstcSchema,
    DstcTurn,
)
import copy
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase

from typing import Any, Optional, Union
import data_prep.data_prep_utils as data_prep_utils
from tod.zs_tod_target import ZsTodTarget
from utilities.dialog_studio_dataclasses import DsDialog, Log
import utils
from utilities.text_utilities import get_nlg_service_name


class KetodNlgApiCallStrategy(DataPrepStrategy):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg)
        self.tod_context_cls = NlgTodContext
        self.tod_turn_cls = KeTodTurn

    def prepare_dialog(
        self,
        ds_dialog: DsDialog,
        schemas: dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> Optional[list[NlgTodTurn]]:
        tod_turns = []
        if not data_prep_utils.is_dialogue_in_domain(
            ds_dialog.services, self.cfg.domains
        ):
            return None
        i = 1
        for ds_turn in ds_dialog.log:
            tod_turn = self.prepare_turn(
                ds_turn, schemas, ds_dialog.services, i, ds_dialog.dialog_index
            )
            i = self.prepare_api_call_turn(turn_csv_row_handler, tod_turn, tod_turns, i)
            i = self.prepare_entity_query_turn(
                turn_csv_row_handler, tod_turn, tod_turns, i
            )
            # i = self.prepare_extra_turns(turn_csv_row_handler, tod_turn, tod_turns, i)
            self.add_tod_turn(
                turn_csv_row_handler, tod_turns, tod_turn, ds_dialog.dialog_index, i
            )
            i += 1
        return tod_turns

    def prepare_turn(self, turn: Log, schemas, services, turn_id: int, dialog_id: int):
        turn_schemas = [schemas[s] for s in services]
        turn_schema_str = self.get_turn_schema_str(turn_schemas)
        context = self.prepare_context(turn)
        target = self.prepare_target(turn, schemas)
        domains = [get_nlg_service_name(s) for s in services]
        return self.tod_turn_cls(
            dialog_id=dialog_id,
            turn_id=turn_id,
            context=context,
            target=target,
            schemas=turn_schemas,
            schema_str=turn_schema_str,
            domains=domains,
            domains_original=services,
        )

    def prepare_context(self, turn: Log) -> KeTodContext:
        context = KeTodContext(
            dialog_history=turn.dialog_history,
            current_user_utterance=turn.user_utterance,
            turn_row_type=TurnRowType.RESPONSE,
        )
        if self.cfg.should_add_service_results:
            for frame in turn.original_system_side_information.frames:
                if not frame.service_results:
                    continue
                if len(frame.service_results) > 0:
                    context.service_results = frame.service_results
                    context.api_call = frame.service_call
        if turn.original_system_side_information.entity_query:
            context.entity_query = turn.original_system_side_information.entity_query
            context.kg_snippets_text = (
                turn.original_system_side_information.kg_snippets_text
            )
        return context

    def prepare_target(self, turn: Log, schemas: dict[str, Any]) -> NlgTodTarget:
        return NlgTodTarget(response=turn.system_response)

    def prepare_entity_query_turn(
        self,
        turn_csv_row_handler: TurnCsvRowBase,
        tod_turn: KeTodTurn,
        tod_turns: list[KeTodTurn],
        turn_id: int,
    ) -> int:
        if not tod_turn.context.entity_query:
            return turn_id
        new_turn = copy.deepcopy(tod_turn)
        new_turn.turn_row_type = TurnRowType.KE_QUERY
        new_turn.context.turn_row_type = TurnRowType.KE_QUERY
        new_turn.target.response = new_turn.context.get_entity_query()
        new_turn.turn_id = turn_id
        new_turn.context.entity_query = None
        new_turn.context.kg_snippets_text = None
        self.add_tod_turn(
            turn_csv_row_handler, tod_turns, new_turn, new_turn.dialog_id, turn_id
        )
        turn_id += 1
        return turn_id

    def prepare_api_call_turn(
        self,
        turn_csv_row_handler: TurnCsvRowBase,
        tod_turn: KeTodTurn,
        tod_turns: list[KeTodTurn],
        turn_id: int,
    ) -> int:
        if not tod_turn.context.api_call:
            return turn_id
        new_turn = copy.deepcopy(tod_turn)
        new_turn.turn_row_type = TurnRowType.API_CALL
        new_turn.context.turn_row_type = TurnRowType.API_CALL
        new_turn.target.response = new_turn.context.get_api_call()
        new_turn.turn_id = turn_id
        new_turn.context.api_call = None
        new_turn.context.service_results = None
        new_turn.context.entity_query = None
        new_turn.context.kg_snippets_text = None
        self.add_tod_turn(
            turn_csv_row_handler, tod_turns, new_turn, new_turn.dialog_id, turn_id
        )
        turn_id += 1
        return turn_id
