from configs.dataprep_config import DataPrepConfig
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.nlg_api_call_strategy import NlgApiCallStrategy
from my_enums import ContextType, DstcSystemActions, SpecialTokens, TurnRowType
from tod.nlg.bitod_api_call import BitodApiCall, BitodApiCallParams
from tod.nlg.bitod_context import BiTodContext
from tod.nlg.ke_tod_turn import KeTodTurn
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.nlg.ke_tod_context import KeTodContext
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


class BitodStrategy(NlgApiCallStrategy):
    def __init__(
        self, cfg: DataPrepConfig, tod_turn_cls=KeTodTurn, tod_context_cls=BiTodContext
    ):
        super().__init__(
            cfg, tod_turn_cls=tod_turn_cls, tod_context_cls=tod_context_cls
        )

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
                    f"Intents: {','.join(intent['name'] for intent in schema.intents)}",
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
        prev_tod_turn = None
        dialog_services = self.get_services(ds_dialog)
        ds_dialog.services = dialog_services
        if not data_prep_utils.is_dialogue_in_domain(dialog_services, self.cfg.domains):
            return None
        i = 1
        for ds_turn in ds_dialog.log:
            tod_turn = self.prepare_turn(
                ds_turn,
                prev_tod_turn,
                schemas,
                dialog_services,
                i,
                ds_dialog.dialog_index,
            )
            tod_turn.is_retrieval = 1 if self.has_request_action(ds_turn) else 0
            tod_turn.is_slot_fill = 1 if self.system_has_request_action(ds_turn) else 0
            i, api_turn = self.prepare_api_call_turn(
                turn_csv_row_handler,
                ds_turn,
                tod_turn,
                prev_tod_turn,
                tod_turns,
                i,
                schemas,
                ds_dialog,
            )
            self.add_tod_turn(
                turn_csv_row_handler,
                tod_turns,
                tod_turn,
                ds_dialog.dialog_index,
                i,
            )
            i += 1
            prev_tod_turn = tod_turn
        return tod_turns

    def prepare_turn(
        self,
        turn: Log,
        prev_tod_turn: KeTodTurn,
        schemas,
        services,
        turn_id: int,
        dialog_id: int,
    ):
        turn_schemas = [schemas[s] for s in services]
        turn_schema_str = self.get_turn_schema_str(turn_schemas)
        context = self.prepare_context(turn, prev_tod_turn)
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

    def prepare_context(self, turn: Log, prev_tod_turn: KeTodTurn) -> BiTodContext:
        if not prev_tod_turn:
            context = self.tod_context_cls(
                max_length=self.cfg.num_turns, context_formatter=self.context_formatter
            )
            context.should_add_sys_actions = self.cfg.should_add_sys_actions
        else:
            context = copy.deepcopy(prev_tod_turn.context)
            context.system_utterances.append(
                prev_tod_turn.context.next_system_utterance
            )
            context.user_utterances.append(context.current_user_utterance)
            context.prev_tod_turn = prev_tod_turn
        context.current_user_utterance = turn.user_utterance
        context.next_system_utterance = turn.system_response
        context.turn_row_type = TurnRowType.RESPONSE.value
        if self.cfg.should_add_service_results:
            if turn.original_system_side_information.PrimaryItem:
                context.service_results = (
                    turn.original_system_side_information.PrimaryItem
                )
                context.api_call = self.get_api_call_from_turn(turn)
        return context

    def prepare_target(self, turn: Log, schemas: dict[str, Any]) -> NlgTodTarget:
        if not turn.system_response:
            return None
        response = self._prepare_response(turn.system_response)
        response += SpecialTokens.eos_token.value
        return NlgTodTarget(response=response)

    def _prepare_response(self, utterance: str) -> str:
        return utterance

    def prepare_api_call_turn(
        self,
        turn_csv_row_handler: TurnCsvRowBase,
        ds_turn: Log,
        tod_turn: KeTodTurn,
        prev_tod_turn: KeTodTurn,
        tod_turns: list[KeTodTurn],
        turn_id: int,
        schemas: dict[str, DstcSchema],
        ds_dialog: DsDialog,
    ) -> int:
        if not tod_turn.context.api_call:
            return turn_id, None
        api_call_response = tod_turn.context.get_api_call()
        copy_ds_turn = copy.deepcopy(ds_turn)
        copy_ds_turn.system_response = api_call_response
        api_call_with_search_results = "\n".join(
            [
                api_call_response,
                "Search Results",
                tod_turn.context.get_service_results(
                    self.cfg.service_results_num_items
                ),
                "End Search Results",
            ]
        )
        tod_turn.context.user_utterances.append(ds_turn.user_utterance)
        tod_turn.context.system_utterances.append(api_call_with_search_results)
        tod_turn.context.current_user_utterance = None

        new_turn = self.prepare_turn(
            copy_ds_turn,
            prev_tod_turn,
            schemas,
            ds_dialog.services,
            turn_id,
            ds_dialog.dialog_index,
        )
        new_turn.context.api_call = None
        new_turn.context.service_results = None
        new_turn.turn_row_type = TurnRowType.API_CALL.value
        new_turn.is_multi_domain_api_call = self.is_multi_domain_api_call(
            new_turn, tod_turns, schemas
        )
        self.add_tod_turn(
            turn_csv_row_handler, tod_turns, new_turn, new_turn.dialog_id, turn_id
        )
        turn_id += 1
        return turn_id, new_turn

    def get_api_call_from_turn(self, turn: Log) -> str:
        if (
            turn.original_system_side_information.PrimaryItem
            and not all(
                [
                    "inform" == action.act
                    for action in turn.original_system_side_information.Actions
                ]
            )
            and not any(
                [
                    "request_more" == action.act
                    for action in turn.original_system_side_information.Actions
                ]
            )
            and not any(
                [
                    "goodbye" == action.act
                    for action in turn.original_system_side_information.Actions
                ]
            )
        ):
            active_intent = turn.original_user_side_information.active_intent
            state = turn.original_user_side_information.state[active_intent]
            delim = "_"
            intent_splits = active_intent.split(delim)
            method = delim.join([intent_splits[0], intent_splits[-1]])
            params = []
            for key, value in state.items():
                if len(value.value) > 1:
                    params.append(
                        BitodApiCallParams(
                            slot_name=key,
                            relation=value.relation,
                            value=f"({', '.join(value.value)})",
                        )
                    )
                else:
                    params.append(
                        BitodApiCallParams(
                            slot_name=key, relation=value.relation, value=value.value[0]
                        )
                    )
            api_call = BitodApiCall(method=method, parameters=params)
            return api_call
        return None
        # return str(api_call)

    def has_request_action(self, ds_turn: Log) -> bool:
        """
        Check if the user turn has a REQUEST action.
        """
        for action in ds_turn.original_user_side_information.Actions:
            if action.act == DstcSystemActions.REQUEST.value.lower():
                return True
        return False

    def system_has_request_action(self, ds_turn: Log) -> bool:
        """
        Check if the system turn has a REQUEST action.
        """
        for action in ds_turn.original_system_side_information.Actions:
            if action.act == DstcSystemActions.REQUEST.value.lower():
                return True
        return False
