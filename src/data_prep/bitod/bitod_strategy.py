from configs.dataprep_config import DataPrepConfig
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.nlg_api_call_data_prep import NlgApiCallDataPrep
from my_enums import TurnRowType
from tod.nlg.bitod_api_call import BitodApiCall, BitodApiCallParams
from tod.nlg.bitod_context import BiTodContext
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
import json


class BitodStrategy(DataPrepStrategy):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg)
        self.tod_context_cls = NlgTodContext
        self.tod_turn_cls = KeTodTurn

    def get_services(self, dialog: DsDialog) -> list[str]:
        data = json.loads(dialog.original_dialog_info)
        services = []
        for key, value in data["Scenario"]["User_Goal"].items():
            name = key.split("_")[0]
            if name not in services:
                services.append(name)
        return services

    def get_turn_schema_str(self, schemas: list[DstcSchema]) -> str:
        out = []
        for schema in schemas:
            schema_str = "\n".join(
                [
                    f"Schema for {schema.service_name}",
                    f"Slots: {','.join(slot['name'] for slot in schema.slots)}",
                ]
            )
            out.append(schema_str)
        return "\n".join(out)

    def prepare_dialog(
        self,
        ds_dialog: DsDialog,
        schemas: dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> Optional[list[NlgTodTurn]]:
        tod_turns = []
        dialog_services = self.get_services(ds_dialog)
        if not data_prep_utils.is_dialogue_in_domain(dialog_services, self.cfg.domains):
            return None
        i = 1
        for ds_turn in ds_dialog.log:
            tod_turn = self.prepare_turn(
                ds_turn,
                schemas,
                dialog_services,
                i,
                ds_dialog.dialog_index,
            )
            i = self.prepare_api_call_turn(turn_csv_row_handler, tod_turn, tod_turns, i)
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
        return self.tod_turn_cls(
            dialog_id=dialog_id,
            turn_id=turn_id,
            context=context,
            target=target,
            schemas=turn_schema_str,
            schema_str=turn_schema_str,
            domains=services,
            domains_original=services,
        )

    def prepare_context(self, turn: Log) -> BiTodContext:
        context = BiTodContext(
            dialog_history=turn.dialog_history,
            current_user_utterance=turn.user_utterance,
            turn_row_type=TurnRowType.RESPONSE.value,
        )
        if self.cfg.should_add_service_results:
            if turn.original_system_side_information.PrimaryItem:
                context.service_results.append(
                    turn.original_system_side_information.PrimaryItem
                )
                context.api_call = self.get_api_call_from_turn(turn)
        return context

    def prepare_target(self, turn: Log, schemas: dict[str, Any]) -> NlgTodTarget:
        return NlgTodTarget(response=turn.system_response)

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
        new_turn.turn_row_type = TurnRowType.API_CALL.value
        new_turn.context.turn_row_type = TurnRowType.API_CALL.value
        new_turn.target.response = new_turn.context.api_call
        new_turn.turn_id = turn_id
        new_turn.context.api_call = None
        new_turn.context.service_results = None
        self.add_tod_turn(
            turn_csv_row_handler, tod_turns, new_turn, new_turn.dialog_id, turn_id
        )
        turn_id += 1
        return turn_id

    def get_api_call_from_turn(self, turn: Log) -> str:
        if not turn.original_system_side_information.PrimaryItem:
            return None
        active_intent = turn.original_user_side_information.active_intent
        state = turn.original_user_side_information.state[active_intent]
        delim = "_"
        intent_splits = active_intent.split(delim)
        method = delim.join([intent_splits[0], intent_splits[-1]])
        params = []
        for key, value in state.items():
            params.append(
                BitodApiCallParams(
                    slot_name=key, relation=value.relation, value=value.value[0]
                )
            )
        api_call = BitodApiCall(method=method, parameters=params)
        return str(api_call)
