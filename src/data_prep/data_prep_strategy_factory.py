from data_prep.bitod.bitod_strategy import BitodStrategy
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.ketod.ketod_nlg_api_call_strategy import KetodNlgApiCallStrategy
from data_prep.nlg_data_prep import NlgDataPrep
from data_prep.nlg_api_call_strategy import NlgApiCallStrategy
from data_prep.zstod_data_prep import ZsTodDataPrep
from my_enums import ContextType
from tod.nlg.pseudo_labels_context import PseudoLabelsContext
from utilities.context_manager import ContextManager


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
        if ContextManager.is_nlg_strategy(context_type):
            if ContextManager.is_sgd_pseudo_labels(context_type):
                return NlgApiCallStrategy(cfg, tod_context_cls=PseudoLabelsContext)
            return NlgApiCallStrategy(cfg)
        if ContextManager.is_ketod(context_type):
            return KetodNlgApiCallStrategy(cfg)
        if ContextManager.is_bitod(context_type):
            return BitodStrategy(cfg)
        # if context_type in [ContextType.GPT_PSEUDO_LABELS.value]:
        #     return cfg
        raise ValueError(f"Unknown data prep step: {context_type}")
