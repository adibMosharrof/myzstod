from abc import ABC, abstractmethod
from my_enums import ContextType
from tod.turns.zs_tod_turn import ZsTodTurn

""" Prepares the csv rows for turns

Extended by [GeneralTurnCsvRow, MultiTaskCsvRow] which can be further extended by [ScaleGradTurnCsvRow]
Provides default headers and row structure for generating csv
Override hook_before_adding_taget to add additional columns to csv
"""


class TurnCsvRowBase(ABC):
    @abstractmethod
    def get_csv_headers(self, should_add_schema: bool = True) -> list[str]:
        headers = ["dialog_id", "turn_id", "domains", "domains_original", "context"]
        if should_add_schema:
            headers.append("schema")
        return headers

    def get_context(self, tod_turn: ZsTodTurn, context_type: ContextType) -> str:
        if context_type not in ContextType.list():
            raise ValueError(
                f"Unknown context type: {context_type}, expected on from {ContextType.list()}"
            )
        if context_type == ContextType.SHORT_REPR:
            return tod_turn.context.get_short_repr()
        return str(tod_turn.context)

    def hook_before_adding_target(self, row: list[str], tod_turn: ZsTodTurn):
        pass

    def to_csv_row(
        self, context_type: ContextType, tod_turn: ZsTodTurn, should_add_schema: bool
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
        target_str = self.get_target_str(tod_turn, context_type)
        row.append(target_str)
        return row

    def get_target_str(
        self, tod_turn: ZsTodTurn, context_type: ContextType = ContextType.DEFAULT
    ) -> str:
        return str(tod_turn.target)
