from data_prep.bitod.bitod_strategy import BitodStrategy
from data_prep.bitod.soloist_bitod_strategy import SoloistBitodStrategy
from data_prep.bitod.zsbitod_data_prep_strategy import ZsBitodDataPrepStrategy
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.ketod.ketod_nlg_api_call_strategy import KetodNlgApiCallStrategy
from data_prep.ketod.soloist_ketod_strategy import SoloistKetodStrategy
from data_prep.ketod.zsketod_data_prep_strategy import ZsKetodDataPrepStrategy
from data_prep.nlg_data_prep import NlgDataPrep
from data_prep.nlg_api_call_strategy import NlgApiCallStrategy
from data_prep.soloist_data_prep import SoloistDataPrep
from data_prep.zstod_api_call_strategy import ZsTodApiCallStrategy
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

        if ContextManager.is_sgd_baseline(context_type):
            baseline_data_prep = (
                ZsTodDataPrep(cfg)
                if ContextManager.is_zs_simple_tod_api_call(context_type)
                else SoloistDataPrep(cfg)
            )
            return ZsTodApiCallStrategy(cfg, baseline_data_prep=baseline_data_prep)
        if ContextManager.is_ketod_baseline(context_type):
            if ContextManager.is_soloist(context_type):
                return SoloistKetodStrategy(cfg)
            return ZsKetodDataPrepStrategy(cfg)

        if ContextManager.is_bitod_baseline(context_type):
            if ContextManager.is_soloist(context_type):
                return SoloistBitodStrategy(cfg)
            return ZsBitodDataPrepStrategy(cfg)

        if ContextManager.is_nlg_strategy(context_type):
            if ContextManager.is_sgd_pseudo_labels(context_type):
                return NlgApiCallStrategy(cfg, tod_context_cls=PseudoLabelsContext)
            return NlgApiCallStrategy(cfg, data_augmentations=cfg.data_augmentations)
        if ContextManager.is_ketod(context_type):
            return KetodNlgApiCallStrategy(cfg)
        if ContextManager.is_bitod(context_type):
            return BitodStrategy(cfg)
        raise ValueError(f"Unknown data prep step: {context_type}")

    @classmethod
    def is_ketod(self, strategy: DataPrepStrategy) -> bool:
        return isinstance(strategy, KetodNlgApiCallStrategy)

    @classmethod
    def is_bitod(self, strategy: DataPrepStrategy) -> bool:
        return isinstance(strategy, BitodStrategy)
