from datamodules.data_augmentation.api_in_context_augmentation import (
    ApiInContextAugmentation,
)
from datamodules.data_augmentation.pseudo_label_augmentation import (
    PseudoLabelAugmentation,
)
from typing import Optional

import numpy as np
import pandas as pd
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
from my_enums import ContextType, DstcSystemActions, TurnRowType, ZsTodConstants
from schema.pseudo_schema_dataclasses import PseudoSchema
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.api_call_turn_csv_row import ApiCallTurnCsvRow
from tod.turns.turn_csv_row_base import TurnCsvRowBase

from utilities.text_utilities import get_nlg_service_name
import utils
import data_prep.data_prep_utils as data_prep_utils
from utilities import text_utilities
from utilities.context_manager import ContextManager


class NlgApiCallStrategy(DataPrepStrategy):
    def __init__(
        self,
        cfg: DataPrepConfig,
        tod_turn_cls=NlgTodTurn,
        tod_context_cls=NlgTodContext,
        data_augmentations=None,
    ):
        super().__init__(
            cfg, tod_turn_cls=tod_turn_cls, tod_context_cls=tod_context_cls
        )
        # self.cfg = cfg
        self.turn_csv_row_cls = ApiCallTurnCsvRow()
        self.data_augmentations = data_augmentations or {}

    def prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, DstcSchema],
    ) -> NlgTodTarget:
        if not system_turn:
            return None
        response = self._prepare_response(system_turn.utterance)
        return NlgTodTarget(response=response)

    def prepare_dialog(
        self,
        dstc_dialog: DstcDialog,
        schemas: dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> Optional[list[NlgTodTurn]]:
        tod_turns = []
        prev_tod_turn = None
        if not data_prep_utils.is_dialogue_in_domain(
            dstc_dialog.services, self.cfg.domains
        ):
            return None
        i = 1
        for user_turn, system_turn in utils.grouper(dstc_dialog.turns, 2):
            tod_turn = self.prepare_turn(
                user_turn, system_turn, prev_tod_turn, schemas, dstc_dialog.services
            )
            if tod_turn.target.response == "":
                continue
            tod_turn.is_retrieval = 1 if self.has_request_action(user_turn) else 0
            tod_turn.is_slot_fill = (
                1 if self.system_has_request_action(system_turn) else 0
            )
            i, api_turn = self.prepare_nlg_api_call_turn(
                turn_csv_row_handler,
                user_turn,
                system_turn,
                tod_turn,
                prev_tod_turn,
                tod_turns,
                schemas,
                dstc_dialog.dialogue_id,
                i,
            )
            if api_turn:
                tod_turn.context.prev_tod_turn = api_turn

            self.add_tod_turn(
                turn_csv_row_handler, tod_turns, tod_turn, dstc_dialog.dialogue_id, i
            )
            i += 1
            prev_tod_turn = tod_turn
        return tod_turns

    def prepare_nlg_api_call_turn(
        self,
        turn_csv_row_handler: TurnCsvRowBase,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        tod_turn: NlgTodTurn,
        prev_tod_turn: NlgTodTurn,
        tod_turns: list[NlgTodTurn],
        schemas: dict[str, DstcSchema],
        dialog_id: int,
        turn_id: int,
    ) -> int:
        if not system_turn:
            return turn_id, None
        if not tod_turn:
            return turn_id, None
        if not tod_turn.context.api_call:
            return turn_id, None
        api_call_response = tod_turn.context.get_api_call(
            schemas=schemas, turn_domains=tod_turn.domains_original
        )
        copy_sys_turn = copy.deepcopy(system_turn)
        copy_sys_turn.utterance = api_call_response
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
        tod_turn.context.system_utterances.append(api_call_with_search_results)
        tod_turn.context.user_utterances.append(user_turn.utterance)
        tod_turn.context.current_user_utterance = None
        # new_turn = copy.deepcopy(tod_turn)
        new_turn = self.prepare_turn(
            user_turn,
            copy_sys_turn,
            prev_tod_turn,
            schemas,
            tod_turn.domains_original,
        )

        new_turn.context.api_call = None
        new_turn.context.service_results = None
        new_turn.turn_row_type = TurnRowType.API_CALL.value
        # new_turn.target.response = tod_turn.context.get_api_call()
        # new_turn.context.next_system_utterance = new_turn.target.response
        new_turn.dialog_id = dialog_id
        new_turn.turn_id = turn_id
        new_turn.is_multi_domain_api_call = self.is_multi_domain_api_call(
            new_turn, tod_turns, schemas
        )
        self.add_tod_turn(turn_csv_row_handler, tod_turns, new_turn, dialog_id, turn_id)
        turn_id += 1
        pseudo_augmentation = self.data_augmentations.get("pseudo_labels", None)
        if pseudo_augmentation:
            aug_turns = pseudo_augmentation.apply(new_turn, tod_turn)
            for aug_turn in aug_turns:
                self.add_tod_turn(
                    turn_csv_row_handler, tod_turns, aug_turn, dialog_id, turn_id
                )
        return turn_id, new_turn

    def is_multi_domain_api_call(self, turn: NlgTodTurn, csv_tod_turns, schemas):
        if len(turn.domains_original) == 1:
            return 0
        df = pd.DataFrame(
            csv_tod_turns, columns=self.turn_csv_row_cls.get_csv_headers()
        )
        api_call_turns = df[df["turn_row_type"] == TurnRowType.API_CALL.value]
        if api_call_turns.empty:
            return 0
        turn_method = text_utilities.get_apicall_method_from_text(turn.target.response)
        turn_domain = self.get_domain_from_api_method_name(
            turn_method, turn.domains_original, schemas
        )
        for _, row in api_call_turns.iterrows():
            prev_turn_method = text_utilities.get_apicall_method_from_text(
                row["target"]
            )
            prev_turn_domain = self.get_domain_from_api_method_name(
                prev_turn_method, row["domains_original"].split(","), schemas
            )
            if prev_turn_domain != turn_domain:
                return 1
        return 0

    def get_domain_from_api_method_name(self, method_name, dialog_domains, schemas):
        en_us = "_en_US"
        for dom in dialog_domains:
            schema = schemas[dom]
            intent_names = self.get_intent_names(schema)
            for intent_name in intent_names:
                if en_us in intent_name:
                    new_intent_name = intent_name.replace(en_us, "")
                    if method_name == new_intent_name:
                        return dom
            if method_name in intent_names:
                return dom
        raise ValueError(f"{method_name} not found in any domain")

    def get_intent_names(self, schema):
        intent_names = []
        field_name = "pseudo_name" if isinstance(schema, PseudoSchema) else "name"
        for intent in schema.intents:
            intent_name = getattr(intent, field_name)
            if not intent_name:
                raise ValueError(
                    f"Intent field {field_name} not found in schema: {schema}"
                )
            intent_names.append(intent_name)
        return intent_names

    def has_request_action(self, user_turn: DstcTurn) -> bool:
        """
        Check if the user turn has a REQUEST action.
        """
        for frame in user_turn.frames:
            for action in frame.actions:
                if action.act == DstcSystemActions.REQUEST.value:
                    return True
        return False

    def system_has_request_action(self, system_turn: DstcTurn) -> bool:
        """
        Check if the system turn has a REQUEST action.
        """
        for frame in system_turn.frames:
            for action in frame.actions:
                if action.act == DstcSystemActions.REQUEST.value:
                    return True
        return False
