


from dataclasses import dataclass
from typing import Optional
from dstc_dataclasses import DstcSchema
from my_enums import SpecialTokens

from simple_tod_dataclasses import MultiTaskSpecialToken
from tod.zs_target import ZsTodTarget
from tod.zs_tod_context import ZsTodContext



@dataclass
class TodTurnCsvRow:
    dialog_id: str
    turn_id: str
    context: str
    target: str = None
    schema: Optional[str] = None


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
class ZsTodTurn:
    context: ZsTodContext
    target: ZsTodTarget
    dialog_id: Optional[str] = None
    turn_id: Optional[int] = None
    schemas: Optional[list[DstcSchema]] = None
    multi_task_token: Optional[MultiTaskSpecialToken] = None
    active_intent: Optional[str] = None
    schema_str: Optional[str] = None
    prompt_token: Optional[SpecialTokens] = None
