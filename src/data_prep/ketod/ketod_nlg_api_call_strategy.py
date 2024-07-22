from configs.dataprep_config import DataPrepConfig
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.nlg_api_call_strategy import NlgApiCallStrategy
from my_enums import ContextType, DstcSystemActions, SpecialTokens, TurnRowType
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

from typing import Any, Optional, Tuple, Union
import data_prep.data_prep_utils as data_prep_utils
from tod.zs_tod_target import ZsTodTarget
from utilities.dialog_studio_dataclasses import DsDialog, Log
import utils
from utilities.text_utilities import get_nlg_service_name


class KetodNlgApiCallStrategy(NlgApiCallStrategy):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg)
        self.tod_context_cls = KeTodContext
        self.tod_turn_cls = KeTodTurn

    def prepare_dialog(
        self,
        ds_dialog: DsDialog,
        schemas: dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> Optional[list[NlgTodTurn]]:
        tod_turns = []
        prev_tod_turn = None
        if not data_prep_utils.is_dialogue_in_domain(
            ds_dialog.services, self.cfg.domains
        ):
            return None
        i = 1
        for ds_turn in ds_dialog.log:
            tod_turn = self.prepare_turn(
                ds_turn,
                prev_tod_turn,
                schemas,
                ds_dialog.services,
                i,
                ds_dialog.dialog_index,
            )
            if tod_turn.target.response == "":
                continue
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
            if api_turn:
                tod_turn.context.prev_tod_turn = api_turn
            i, entity_query_turn = self.prepare_entity_query_turn(
                turn_csv_row_handler,
                ds_turn,
                tod_turn,
                prev_tod_turn,
                tod_turns,
                i,
                schemas,
                ds_dialog,
            )
            if entity_query_turn:
                if api_turn:
                    entity_query_turn.context.prev_tod_turn = api_turn
                    tod_turn.context.prev_tod_turn = entity_query_turn
                else:
                    tod_turn.context.prev_tod_turn = entity_query_turn
            # i = self.prepare_extra_turns(turn_csv_row_handler, tod_turn, tod_turns, i)
            self.add_tod_turn(
                turn_csv_row_handler, tod_turns, tod_turn, ds_dialog.dialog_index, i
            )
            i += 1
            prev_tod_turn = tod_turn
        return tod_turns

    def prepare_entity_query_turn(
        self,
        turn_csv_row_handler: TurnCsvRowBase,
        ds_turn: Log,
        tod_turn: KeTodTurn,
        prev_tod_turn: KeTodTurn,
        tod_turns: list[KeTodTurn],
        turn_id: int,
        schemas: dict[str, DstcSchema],
        ds_dialog: DsDialog,
    ) -> Tuple[int, Optional[KeTodTurn]]:
        if not ds_turn.original_system_side_information.entity_query:
            return turn_id, None

        entity_query_response = tod_turn.context.get_entity_query()
        copy_ds_turn = copy.deepcopy(ds_turn)
        copy_ds_turn.system_response = entity_query_response
        api_call_with_kg_snippets_text = "\n".join(
            [
                entity_query_response,
                tod_turn.context.get_kg_snippets_text(),
            ]
        )
        # tod_turn.context.dialog_history += f"<USER> {ds_turn.user_utterance}"
        # tod_turn.context.dialog_history += f"<SYSTEM> {api_call_with_kg_snippets_text}"
        tod_turn.context.user_utterances.append(ds_turn.user_utterance)
        tod_turn.context.system_utterances.append(api_call_with_kg_snippets_text)
        tod_turn.context.current_user_utterance = None

        new_turn = self.prepare_turn(
            copy_ds_turn,
            prev_tod_turn,
            schemas,
            ds_dialog.services,
            turn_id,
            ds_dialog.dialog_index,
        )
        new_turn.turn_row_type = TurnRowType.KE_QUERY.value
        new_turn.context.api_call = None
        new_turn.context.service_results = None
        # new_turn.turn_row_type = TurnRowType.API_CALL.value

        new_turn.context.entity_query = None
        new_turn.context.kg_snippets_text = None
        self.add_tod_turn(
            turn_csv_row_handler, tod_turns, new_turn, new_turn.dialog_id, turn_id
        )
        turn_id += 1
        return turn_id, new_turn

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
    ) -> Tuple[int, Optional[KeTodTurn]]:
        if not ds_turn.original_system_side_information.frames[0].service_call:
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
        # tod_turn.context.dialog_history += f"<USER> {ds_turn.user_utterance}"
        # tod_turn.context.dialog_history += f"<SYSTEM> {api_call_with_search_results}"
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
        new_turn.context.entity_query = None
        new_turn.context.kg_snippets_text = None
        self.add_tod_turn(
            turn_csv_row_handler, tod_turns, new_turn, new_turn.dialog_id, turn_id
        )
        turn_id += 1
        return turn_id, new_turn

    def prepare_turn(
        self,
        turn: Log,
        prev_tod_turn: KeTodTurn,
        schemas,
        services,
        turn_id: int,
        dialog_id: int,
    ) -> KeTodTurn:
        turn_schemas = [schemas[s] for s in services]
        turn_schema_str = self.get_turn_schema_str(turn_schemas)
        context = self.prepare_context(turn, prev_tod_turn)
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

    def prepare_context(self, turn: Log, prev_tod_turn: KeTodTurn) -> KeTodContext:
        if not prev_tod_turn:
            context = self.tod_context_cls(max_length=self.cfg.num_turns)
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
            for frame in turn.original_system_side_information.frames:
                context.service_results = frame.service_results
                context.api_call = frame.service_call
        if turn.original_system_side_information.entity_query:
            context.entity_query = turn.original_system_side_information.entity_query
            context.kg_snippets_text = (
                turn.original_system_side_information.kg_snippets_text
            )
        return context

    def prepare_target(self, turn: Log, schemas: dict[str, Any]) -> NlgTodTarget:
        if not turn.system_response:
            return None
        response = self._prepare_response(turn.system_response)
        return NlgTodTarget(response=response)
    
    def _prepare_response(self, utterance:str) -> str:
        if self.cfg.context_type == ContextType.KETOD_GPT_API_CALL.value:
            utterance += SpecialTokens.eos_token.value
        return utterance

    def has_request_action(self, log: Log) -> bool:
        """
        Check if the user turn has a REQUEST action.
        """
        for frame in log.original_user_side_information.frames:
            for action in frame.actions:
                if action.act == DstcSystemActions.REQUEST.value:
                    return True
        return False

    def system_has_request_action(self, log: Log) -> bool:
        """
        Check if the system turn has a REQUEST action.
        """
        for frame in log.original_system_side_information.frames:
            for action in frame.actions:
                if action.act == DstcSystemActions.REQUEST.value:
                    return True
        return False
