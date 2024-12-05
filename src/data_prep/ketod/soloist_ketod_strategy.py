from configs.dataprep_config import DataPrepConfig
from data_prep.ketod.zsketod_data_prep_strategy import ZsKetodDataPrepStrategy
from utilities.dialog_studio_dataclasses import Log


class SoloistKetodStrategy(ZsKetodDataPrepStrategy):

    # def __init__(self, cfg: DataPrepConfig):
    #     super().__init__(cfg)

    def _get_actions(self, turn: Log) -> None:
        return None
