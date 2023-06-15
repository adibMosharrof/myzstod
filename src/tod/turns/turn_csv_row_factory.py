from configs.dataprep_config import DataPrepConfig
from tod.turns.general_turn_csv_row import GeneralTurnCsvRow
from tod.turns.mh_turn_csv_row import MhTurnCsvRow
from tod.turns.multi_turn_csv_row import MultiTaskCsvRow
from tod.turns.turn_csv_row import TurnCsvRowBase


class TurnCsvRowFactory:
    @classmethod
    def get_handler(self, cfg: DataPrepConfig) -> TurnCsvRowBase:
        if cfg.is_multi_head:
            return MhTurnCsvRow(cfg.mh_fact)
        if cfg.is_multi_task:
            return MultiTaskCsvRow()
        return GeneralTurnCsvRow()
