from configs.dataprep_config import DataPrepConfig
from my_enums import ContextType
from tod.turns.bitod_csv_row import BitodCsvRow
from tod.turns.general_turn_csv_row import GeneralTurnCsvRow
from tod.turns.ketod_api_call_turn_csv_row import KetodApiCallTurnCsvRow
from tod.turns.mh_turn_csv_row import MhTurnCsvRow
from tod.turns.multi_task_csv_row import MultiTaskCsvRow
from tod.turns.scalegrad_turn_csv_row import ScaleGradTurnCsvRow
from tod.turns.api_call_turn_csv_row import ApiCallTurnCsvRow
from tod.turns.turn_csv_row_base import TurnCsvRowBase


class TurnCsvRowFactory:
    @classmethod
    def get_handler(self, cfg: DataPrepConfig) -> TurnCsvRowBase:
        csv_row_cls = GeneralTurnCsvRow()
        if cfg.is_multi_task:
            return MultiTaskCsvRow()
        if cfg.is_scale_grad:
            return ScaleGradTurnCsvRow(csv_row_cls)
        if cfg.context_type in [
            ContextType.NLG_API_CALL.value,
            ContextType.GPT_API_CALL.value,
            ContextType.GPT_CROSS.value,
        ]:
            return ApiCallTurnCsvRow()
        if cfg.context_type in [
            ContextType.KETOD_API_CALL.value,
            ContextType.KETOD_GPT_API_CALL.value,
        ]:
            return KetodApiCallTurnCsvRow()
        if cfg.context_type in [
            ContextType.BITOD_GPT.value,
            ContextType.BITOD.value,
        ]:
            return BitodCsvRow()
        return csv_row_cls
