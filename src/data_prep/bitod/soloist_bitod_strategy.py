from data_prep.bitod.zsbitod_data_prep_strategy import ZsBitodDataPrepStrategy
from utilities.dialog_studio_dataclasses import Log


class SoloistBitodStrategy(ZsBitodDataPrepStrategy):

    def _get_actions(self, turn: dict, user_turn: dict) -> None:
        return None
