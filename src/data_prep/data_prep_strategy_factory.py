from data_prep.bitod.bitod_strategy import BitodStrategy
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.ketod.ketod_nlg_api_call_strategy import KetodNlgApiCallStrategy
from data_prep.nlg_data_prep import NlgDataPrep
from data_prep.nlg_api_call_strategy import NlgApiCallStrategy
from data_prep.zstod_data_prep import ZsTodDataPrep
from my_enums import ContextType


class DataPrepStrategyFactory:
    """
    Class for resolving data preparation strategy.
    """

    @classmethod
    def get_strategy(cls, cfg, context_type: ContextType) -> DataPrepStrategy:
        """
        Resolves data preparation strategy.
        """
        if context_type == ContextType.NLG.value:
            return NlgDataPrep(cfg)
        if context_type == ContextType.SHORT_REPR.value:
            return ZsTodDataPrep(cfg)
        if context_type in [
            ContextType.NLG_API_CALL.value,
            ContextType.GPT_API_CALL.value,
            ContextType.GPT_CROSS.value,
        ]:
            return NlgApiCallStrategy(cfg)
        if context_type in [
            ContextType.KETOD_API_CALL.value,
            ContextType.KETOD_GPT_API_CALL.value,
        ]:
            return KetodNlgApiCallStrategy(cfg)
        if context_type in [
            ContextType.BITOD.value,
            ContextType.BITOD_GPT.value,
        ]:
            return BitodStrategy(cfg)
        raise ValueError(f"Unknown data prep step: {context_type}")
