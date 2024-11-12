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
    SpecialTokens,
    TurnRowType,
    ZsTodConstants,
)
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
from data_prep.base_api_call_strategy import BaseApiCallStrategy


class NlgApiCallStrategy(BaseApiCallStrategy):
    def __init__(
        self,
        cfg: DataPrepConfig,
        tod_turn_cls=NlgTodTurn,
        tod_context_cls=NlgTodContext,
        data_augmentations=None,
    ):
        super().__init__(
            cfg,
            tod_turn_cls=tod_turn_cls,
            tod_context_cls=tod_context_cls,
            data_augmentations=data_augmentations,
        )

    def prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, DstcSchema],
    ) -> NlgTodTarget:
        if not system_turn:
            return None
        response = self._prepare_response(system_turn.utterance)
        response += SpecialTokens.eos_token.value
        return NlgTodTarget(response=response)
