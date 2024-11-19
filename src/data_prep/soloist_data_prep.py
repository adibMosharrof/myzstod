from configs.dataprep_config import DataPrepConfig
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
    DstcTurn,
)
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.zstod_data_prep import ZsTodDataPrep
from my_enums import ZsTodConstants
from tod.turns.zs_tod_turn import ZsTodTurn
from tod.zs_tod_dst import ZsTodDst
from tod.zs_tod_target import ZsTodTarget
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief
from tod.zs_tod_context import ZsTodContext


class SoloistDataPrep(ZsTodDataPrep):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg)

    def _get_actions(self, turn: DstcTurn) -> None:
        return None
