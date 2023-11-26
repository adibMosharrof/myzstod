from dataclasses import dataclass
from typing import Optional

from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from sgd_dstc8_data_model.dstc_dataclasses import DstcSchema


@dataclass
class NlgTodTurn:
    context: NlgTodContext
    target: NlgTodTarget
    schemas: list[DstcSchema]
    schema_str: str
    domains: list[str]
    domains_original: list[str]
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None
    is_service_call: Optional[bool] = False
