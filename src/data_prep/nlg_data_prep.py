from configs.dataprep_config import DataPrepConfig
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
    DstcTurn,
)
from data_prep.data_prep_strategy import DataPrepStrategy
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from tod.nlg.nlg_tod_turn import NlgTodTurn


class NlgDataPrep(DataPrepStrategy):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg, tod_turn_cls=NlgTodTurn, tod_context_cls=NlgTodContext)
        # self.cfg = cfg

    def prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, DstcSchema],
    ) -> NlgTodTarget:
        response = self._prepare_response(system_turn)
        return NlgTodTarget(response=response)

    def get_turn_schema_str(self, turn_schemas) -> str:
        return "".join([s.get_nlg_repr() for s in turn_schemas])
