from data_prep.base_api_call_strategy import BaseApiCallStrategy
from data_prep.zstod_data_prep import ZsTodDataPrep
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
from my_enums import (
    ContextType,
    DstcSystemActions,
    TurnRowType,
    ZsTodConstants,
    ZsTodSystemActions,
)
from schema.pseudo_schema_dataclasses import PseudoSchema
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.api_call_turn_csv_row import ApiCallTurnCsvRow
from tod.turns.turn_csv_row_base import TurnCsvRowBase

from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_context import ZsTodContext
from tod.zs_tod_target import ZsTodTarget
from utilities.text_utilities import get_nlg_service_name
import utils
import data_prep.data_prep_utils as data_prep_utils
from utilities import text_utilities
from utilities.context_manager import ContextManager


class ZsTodApiCallStrategy(BaseApiCallStrategy):
    def __init__(
        self,
        cfg: DataPrepConfig,
        tod_turn_cls=NlgTodTurn,
        tod_context_cls=ZsTodContext,
        data_augmentations=None,
    ):
        super().__init__(
            cfg,
            tod_turn_cls=tod_turn_cls,
            tod_context_cls=tod_context_cls,
            data_augmentations=data_augmentations,
        )
        self.zstod_data_prep = ZsTodDataPrep(cfg)

    def prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, DstcSchema],
    ) -> ZsTodTarget:
        return self.zstod_data_prep.prepare_target(user_turn, system_turn, schemas)

    def get_turn_schema_str(self, turn_schemas) -> str:
        return self.zstod_data_prep.get_turn_schema_str(turn_schemas)

    def get_api_call_turn(
        self,
        user_turn,
        tod_turn,
        prev_tod_turn,
        tod_turns,
        schemas,
        dialog_id,
        turn_id,
        copy_sys_turn,
    ):
        api_call_turn = super().get_api_call_turn(
            user_turn,
            tod_turn,
            prev_tod_turn,
            tod_turns,
            schemas,
            dialog_id,
            turn_id,
            copy_sys_turn,
        )
        try:
            domain_name = tod_turn.target.actions[0].domain
        except:
            domain_name = ""
        api_call_turn.target.actions = [
            ZsTodAction(
                domain=domain_name,
                action_type=ZsTodSystemActions.API_CALL.value,
            )
        ]
        return api_call_turn
