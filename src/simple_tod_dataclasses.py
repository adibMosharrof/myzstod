from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import DefaultDict, Dict, List, Optional, Tuple, Union
from dotmap import DotMap
import numpy as np
import pandas as pd

from torch import nn
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcRequestedSlot,
    DstcSchema,
    DstcServiceCall,
)
from multi_head.mh_dataclasses import MultiHeadDictFactory, MultiHeadInstance

from my_enums import (
    ContextType,
    DstcSystemActions,
    MultiTaskNames,
    SpecialTokens,
)
import dstc.dstc_utils as dstc_utils


@dataclass
class MultiTaskSpecialToken:
    start_tokens: list[SpecialTokens]
    end_tokens: list[SpecialTokens]
    prompt_token: SpecialTokens
    name: MultiTaskNames


# Datamodule classes


@dataclass
class TodTestDataBatch:
    input_ids: list[list[int]]
    attention_masks: list[list[int]]
    contexts_text: list[str]
    targets_text: list[str]
    dialog_ids: list[int]
    turn_ids: list[int]
    turn_row_types: Optional[list[bool]] = field(default_factory=list)


@dataclass
class NlgTestDataBatch:
    input_ids: list[list[int]]
    attention_masks: list[list[int]]
    labels: list[list[int]]
    dialog_ids: list[int]
    turn_ids: list[int]
    domain_ids: list[list[int]]
    turn_row_types: Optional[list[bool]] = field(default_factory=list)
    is_retrievals: Optional[list[int]] = field(default_factory=list)
    is_slot_fills: Optional[list[int]] = field(default_factory=list)
    is_multi_domain_api_calls: Optional[list[int]] = field(default_factory=list)


@dataclass
class CrossTestDataBatch(NlgTestDataBatch):
    encoder_hidden_states: list[list[int]] = field(default_factory=list)


@dataclass
class PredRef:
    pred: str
    ref: str


@dataclass(frozen=True, eq=True)
class MultiTaskTurnKey:
    dialog_id: str
    turn_id: int


class InferenceRecords:
    def __init__(self):
        self.preds = []
        self.dialog_ids = []
        self.turn_ids = []
        self.refs = []
        self.contexts = []
        self.is_data_concatenated = False

    def add(self, preds, refs, dialog_ids, turn_ids, contexts):
        self.preds.append(preds)
        self.dialog_ids.append(dialog_ids)
        self.turn_ids.append(turn_ids)
        self.refs.append(refs)
        self.contexts.append(contexts)

    def concat_data(self):
        self.preds = np.concatenate(self.preds, axis=0)
        self.refs = np.concatenate(self.refs, axis=0)
        self.dialog_ids = np.concatenate(self.dialog_ids, axis=0)
        self.turn_ids = np.concatenate(self.turn_ids, axis=0)
        self.contexts = np.concatenate(self.contexts, axis=0)
        self.is_data_concatenated = True

    def get_data_by_turns(self) -> Dict[MultiTaskTurnKey, list[PredRef]]:
        if not self.is_data_concatenated:
            self.concat_data()
        turns: DefaultDict[MultiTaskTurnKey, list[PredRef]] = defaultdict(list)
        for pred, ref, dialog_id, turn_id in zip(
            self.preds, self.refs, self.dialog_ids, self.turn_ids
        ):
            turns[MultiTaskTurnKey(dialog_id, turn_id)].append(PredRef(pred, ref))
        return turns

    def get_data_for_multitask(self) -> Tuple[list[str], list[str]]:
        turns = self.get_data_by_turns()
        preds: list[str] = []
        refs: list[str] = []
        for key, turn in turns.items():
            mt_preds = "".join(
                [
                    SpecialTokens.begin_target,
                    "".join([self.extract_target(t.pred) for t in turn]),
                    SpecialTokens.end_target,
                ]
            )
            mt_refs = "".join(
                [
                    SpecialTokens.begin_target,
                    "".join([self.extract_target(t.ref) for t in turn]),
                    SpecialTokens.end_target,
                ]
            )
            preds.append(mt_preds)
            refs.append(mt_refs)
        return preds, refs

    def extract_target(self, text: str) -> str:
        return dstc_utils.remove_tokens_from_text(
            text,
            [
                SpecialTokens.begin_target,
                SpecialTokens.end_target,
                SpecialTokens.bos_token,
                SpecialTokens.eos_token,
            ],
        )


def get_multi_task_special_tokens() -> dict[str, MultiTaskSpecialToken]:
    return {
        MultiTaskNames.DSTS.value: MultiTaskSpecialToken(
            [SpecialTokens.begin_dsts],
            [SpecialTokens.end_dsts],
            SpecialTokens.prompt_dst,
            MultiTaskNames.DSTS,
        ),
        MultiTaskNames.ACTIONS.value: MultiTaskSpecialToken(
            [SpecialTokens.begin_user_action, SpecialTokens.begin_action],
            [SpecialTokens.end_user_action, SpecialTokens.end_action],
            SpecialTokens.prompt_action,
            MultiTaskNames.ACTIONS,
        ),
        MultiTaskNames.NLG.value: MultiTaskSpecialToken(
            [SpecialTokens.begin_response],
            [SpecialTokens.end_response],
            SpecialTokens.prompt_response,
            MultiTaskNames.NLG,
        ),
    }

    return [
        MultiTaskSpecialToken(
            [SpecialTokens.begin_dsts],
            [SpecialTokens.end_dsts],
            SpecialTokens.prompt_dst,
            MultiTaskNames.DSTS,
        ),
        MultiTaskSpecialToken(
            [SpecialTokens.begin_user_action, SpecialTokens.begin_action],
            [SpecialTokens.end_user_action, SpecialTokens.end_action],
            SpecialTokens.prompt_action,
            MultiTaskNames.ACTIONS,
        ),
        MultiTaskSpecialToken(
            [SpecialTokens.begin_response],
            [SpecialTokens.end_response],
            SpecialTokens.prompt_response,
            MultiTaskNames.NLG,
        ),
    ]
