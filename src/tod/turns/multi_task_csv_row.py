from my_enums import ContextType
from tod.turns.turn_csv_row import TurnCsvRowBase
from tod.turns.zs_tod_turn import ZsTodTurn


class MultiTaskCsvRow(TurnCsvRowBase):
    def get_csv_headers(self, should_add_schema: bool) -> list[str]:
        headers = super().get_csv_headers(should_add_schema)
        headers += ["task", "target"]
        return headers

    def to_csv_row(
        self, context_type: ContextType, tod_turn: ZsTodTurn, should_add_schema: bool
    ) -> list[str]:
        row = super().to_csv_row(context_type, tod_turn, should_add_schema)
        target_str = str(tod_turn.target)
        row += [tod_turn.multi_task_token.name.value, target_str]

        return row

    def hook_before_adding_target(self, row: list[str], tod_turn: ZsTodTurn):
        row.append(tod_turn.multi_task_token.name.value)
