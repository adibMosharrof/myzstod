from dataclasses import dataclass
from typing import Optional, Union

from tod.nlg.ke_tod_context import KeTodContext
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from sgd_dstc8_data_model.dstc_dataclasses import DstcSchema
from my_enums import TurnRowType


@dataclass
class NlgTodTurn:
    context: Union[NlgTodContext, KeTodContext]
    target: NlgTodTarget
    schemas: list[DstcSchema]
    schema_str: str
    domains: list[str]
    domains_original: list[str]
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None
    turn_row_type: Optional[TurnRowType] = TurnRowType.RESPONSE.value
    is_retrieval: Optional[int] = 0
    is_slot_fill: Optional[int] = 0
    is_multi_domain_api_call: Optional[int] = 0
    dataset_name: Optional[str] = None
