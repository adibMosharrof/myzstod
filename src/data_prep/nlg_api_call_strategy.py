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
        response += SpecialTokens.eos_token.value
        return NlgTodTarget(response=response)

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
