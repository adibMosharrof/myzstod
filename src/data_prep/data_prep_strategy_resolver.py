from data_prep.nlg_data_prep import NlgDataPrep
from data_prep.nlg_service_call_data_prep import NlgServiceCallDataPrep
from data_prep.zstod_data_prep import ZsTodDataPrep
from my_enums import ContextType


class DataPrepStrategyResolver:
    """
    Class for resolving data preparation strategy.
    """

    @classmethod
    def resolve(self, cfg):
        """
        Resolves data preparation strategy.
        """
        if cfg.context_type == ContextType.NLG.value:
            return NlgDataPrep(cfg)
        if cfg.context_type == ContextType.SHORT_REPR.value:
            return ZsTodDataPrep(cfg)
        if cfg.context_type == ContextType.NLG_SERVICE_CALL.value:
            return NlgServiceCallDataPrep(cfg)
        raise ValueError(f"Unknown data prep step: {cfg.context_type}")
