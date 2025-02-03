from dataclasses import dataclass, fields
from typing import Optional
from sgd_dstc8_data_model.dstc_dataclasses import DstcSchema
from my_enums import ContextType, SpecialTokens

from simple_tod_dataclasses import MultiTaskSpecialToken
from tod.zs_tod_target import ZsTodTarget
from tod.zs_tod_context import ZsTodContext
from utilities.context_manager import ContextManager


@dataclass
class TodTurnCsvRow:
    dialog_id: str
    turn_id: str
    context: str
    target: str = None
    schema: Optional[str] = None
    domains: Optional[str] = None
    domains_original: Optional[str] = None


@dataclass
class TodTurnMultiHeadCsvRow:
    dialog_id: str
    turn_id: str
    context: str
    user_actions: Optional[str] = ""
    system_actions: Optional[str] = ""
    dsts: Optional[str] = ""
    nlg: Optional[str] = ""
    schema: Optional[str] = ""

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class TodTurnMultiTaskCsvRow(TodTurnCsvRow):
    task: Optional[str] = ""


@dataclass
class TodTurnScaleGradCsvRow(TodTurnMultiTaskCsvRow):
    special_tokens: Optional[str] = ""


@dataclass
class TodTurnApiCallCsvRow(TodTurnCsvRow):

    turn_row_type: Optional[int] = None
    is_retrieval: Optional[int] = None
    is_slot_fill: Optional[int] = None
    is_multi_domain_api_call: Optional[int] = None
    dataset_name: Optional[str] = None
    is_single_domain: Optional[int] = None
    current_user_utterance: Optional[str] = None

    @classmethod
    def from_list_of_values_and_headers(self, values, headers):
        header_value_map = dict(zip(headers, values))
        ordered_values = [header_value_map[field.name] for field in fields(self)]
        return self(*ordered_values)


@dataclass
class ZsTodTurn:
    context: ZsTodContext
    target: ZsTodTarget
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None
    schemas: Optional[list[DstcSchema]] = None
    multi_task_token: Optional[MultiTaskSpecialToken] = None
    active_intent: Optional[str] = None
    schema_str: Optional[str] = None
    prompt_token: Optional[SpecialTokens] = ""
    domains: Optional[list[str]] = None
    domains_original: Optional[list[str]] = None


@dataclass
class ZsTodApiTurn(ZsTodTurn):
    turn_row_type: Optional[int] = None
    is_retrieval: Optional[int] = None
    is_slot_fill: Optional[int] = None
    is_multi_domain_api_call: Optional[int] = None
    dataset_name: Optional[str] = None


class TodTurnCsvRowFactory:
    @classmethod
    def get_handler(self, cfg):
        if cfg.get("is_scale_grad", 0):
            return TodTurnScaleGradCsvRow
        if any(
            [
                ContextManager.is_nlg_strategy(cfg.model_type.context_type),
                ContextManager.is_ketod(cfg.model_type.context_type),
                ContextManager.is_bitod(cfg.model_type.context_type),
            ]
        ):
            return TodTurnApiCallCsvRow
        if any(
            [
                ContextManager.is_zstod(cfg.model_type.context_type),
                ContextManager.is_simple_tod(cfg.model_type.context_type),
            ]
        ):
            return TodTurnCsvRow
        raise ValueError("incorrect context type")
        return TodTurnCsvRow
