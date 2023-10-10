from configs.dataprep_config import DataPrepConfig
from tod.turns.general_turn_csv_row import GeneralTurnCsvRow
from tod.turns.mh_turn_csv_row import MhTurnCsvRow
from tod.turns.multi_task_csv_row import MultiTaskCsvRow
from tod.turns.scalegrad_turn_csv_row import ScaleGradTurnCsvRow
from tod.turns.turn_csv_row_base import TurnCsvRowBase


class TurnCsvRowFactory:
    @classmethod
    def get_handler(self, cfg: DataPrepConfig) -> TurnCsvRowBase:
        csv_row_cls = GeneralTurnCsvRow()
        if cfg.is_multi_task:
            csv_row_cls = MultiTaskCsvRow()
        if cfg.is_scale_grad:
            csv_row_cls = ScaleGradTurnCsvRow(csv_row_cls)
        return csv_row_cls
