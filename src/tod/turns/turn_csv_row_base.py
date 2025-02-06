from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from my_enums import ContextType, Steps

if TYPE_CHECKING:
    from tod.turns.zs_tod_turn import ZsTodTurn
from utilities.context_manager import ContextManager

""" Prepares the csv rows for turns

Extended by [GeneralTurnCsvRow, MultiTaskCsvRow] which can be further extended by [ScaleGradTurnCsvRow]
Provides default headers and row structure for generating csv
Override hook_before_adding_taget to add additional columns to csv
"""


@dataclass
class TurnCsvRowBase(ABC):
    dialog_id: str = None
    turn_id: str = None
    context: str = None
    target: str = None
    schema: Optional[str] = None
    domains: Optional[str] = None
    domains_original: Optional[str] = None

    @abstractmethod
    def get_csv_headers(self, should_add_schema: bool = True) -> list[str]:
        headers = ["dialog_id", "turn_id", "domains", "domains_original", "context"]
        if should_add_schema:
            headers.append("schema")
        return headers

    def get_context(self, tod_turn: "ZsTodTurn", context_type: ContextType) -> str:
        if context_type not in ContextType.list():
            raise ValueError(
                f"Unknown context type: {context_type}, expected on from {ContextType.list()}"
            )
        if context_type == ContextType.SHORT_REPR:
            return tod_turn.context.get_short_repr()
        return str(tod_turn.context)

    def hook_before_adding_target(self, row: list[str], tod_turn: "ZsTodTurn"):
        pass

    def to_csv_row(
        self,
        context_type: ContextType,
        tod_turn: "ZsTodTurn",
        should_add_schema: bool,
        step_name=None,
    ) -> list[str]:
        context_str = self.get_context(tod_turn, context_type)

        context_str += getattr(tod_turn, "prompt_token", "")
        # context_str += tod_turn.prompt_token if tod_turn.prompt_token else ""
        domains_str = ",".join(tod_turn.domains) if tod_turn.domains else ""
        domains_orig_str = (
            ",".join(tod_turn.domains_original) if tod_turn.domains else ""
        )
        row = [
            tod_turn.dialog_id,
            tod_turn.turn_id,
            domains_str,
            domains_orig_str,
            context_str,
        ]
        if should_add_schema:
            row.append(tod_turn.schema_str)
        self.hook_before_adding_target(row, tod_turn)
        target_str = self.get_target_str(tod_turn, context_type, step_name)
        row.append(target_str)
        return row

    def get_target_str(
        self,
        tod_turn: "ZsTodTurn",
        context_type: ContextType = ContextType.DEFAULT,
        step_name=None,
    ) -> str:
        if not step_name == Steps.TEST.value:
            return str(tod_turn.target)
        if any(
            [
                ContextManager.is_sgd_baseline(context_type),
                ContextManager.is_ketod_baseline(context_type),
            ]
        ):
            return tod_turn.target.get_nlg_target_str()
        return str(tod_turn.target)
