from my_enums import ContextType


class ContextManager:
    @staticmethod
    def is_ketod(context_type: str) -> bool:
        return context_type in [
            ContextType.KETOD_API_CALL.value,
            ContextType.KETOD_GPT_API_CALL.value,
            ContextType.KETOD_SOLOIST_API_CALL.value,
            ContextType.KETOD_ZSTOD_API_CALL.value,
            ContextType.KETOD_SIMPLE_TOD_API_CALL.value,
        ]

    @staticmethod
    def is_bitod(context_type: str) -> bool:
        return context_type in [
            ContextType.BITOD.value,
            ContextType.BITOD_GPT.value,
            ContextType.BITOD_SIMPLE_TOD_API_CALL.value,
            ContextType.BITOD_SOLOIST_API_CALL.value,
            ContextType.BITOD_ZSTOD_API_CALL.value,
        ]

    @staticmethod
    def is_sgd_nlg_api(context_type: str) -> bool:
        return context_type in [
            ContextType.NLG_API_CALL.value,
            ContextType.GPT_API_CALL.value,
            ContextType.GPT_CROSS.value,
        ]

    @staticmethod
    def is_nlg_strategy(context_type: str) -> bool:
        return context_type in [
            ContextType.NLG_API_CALL.value,
            ContextType.GPT_API_CALL.value,
            ContextType.GPT_CROSS.value,
            ContextType.GPT_PSEUDO_LABELS.value,
            ContextType.ZSTOD_API_CALL.value,
            ContextType.SIMPLE_TOD_API_CALL.value,
            ContextType.SOLOIST_API_CALL.value,
        ]

    @staticmethod
    def is_sgd_pseudo_labels(context_type: str) -> bool:
        return context_type in [ContextType.GPT_PSEUDO_LABELS.value]

    @staticmethod
    def is_decoder_type(context_type: str) -> bool:
        return context_type in [
            ContextType.GPT_API_CALL.value,
            ContextType.GPT_CROSS.value,
            ContextType.GPT_PSEUDO_LABELS.value,
            ContextType.BITOD_GPT.value,
            ContextType.KETOD_GPT_API_CALL.value,
        ]

    @staticmethod
    def is_zstod(context_type: str) -> bool:
        return context_type in [
            ContextType.SHORT_REPR.value,
            ContextType.ZSTOD_API_CALL.value,
            ContextType.KETOD_ZSTOD_API_CALL.value,
            ContextType.BITOD_ZSTOD_API_CALL.value,
        ]

    @staticmethod
    def is_simple_tod(context_type: str) -> bool:
        return context_type in [
            ContextType.DEFAULT.value,
            ContextType.SIMPLE_TOD_API_CALL.value,
            ContextType.KETOD_SIMPLE_TOD_API_CALL.value,
            ContextType.BITOD_SIMPLE_TOD_API_CALL.value,
        ]

    @staticmethod
    def is_soloist(context_type: str) -> bool:
        return context_type in [
            ContextType.SOLOIST_API_CALL.value,
            ContextType.KETOD_SOLOIST_API_CALL.value,
            ContextType.BITOD_SOLOIST_API_CALL.value,
        ]

    @staticmethod
    def is_zs_simple_tod_api_call(context_type: str) -> bool:
        return context_type in [
            ContextType.ZSTOD_API_CALL.value,
            ContextType.SIMPLE_TOD_API_CALL.value,
            ContextType.KETOD_ZSTOD_API_CALL.value,
            ContextType.KETOD_SIMPLE_TOD_API_CALL.value,
            ContextType.BITOD_ZSTOD_API_CALL.value,
            ContextType.BITOD_SIMPLE_TOD_API_CALL.value,
        ]

    @staticmethod
    def is_sgd_baseline(context_type: str) -> bool:
        return context_type in [
            ContextType.SIMPLE_TOD_API_CALL.value,
            ContextType.ZSTOD_API_CALL.value,
            ContextType.SOLOIST_API_CALL.value,
        ]

    @staticmethod
    def is_ketod_baseline(context_type: str) -> bool:
        return context_type in [
            ContextType.KETOD_SIMPLE_TOD_API_CALL.value,
            ContextType.KETOD_ZSTOD_API_CALL.value,
            ContextType.KETOD_SOLOIST_API_CALL.value,
        ]

    @staticmethod
    def is_bitod_baseline(context_type: str) -> bool:
        return context_type in [
            ContextType.BITOD_SIMPLE_TOD_API_CALL.value,
            ContextType.BITOD_SOLOIST_API_CALL.value,
            ContextType.BITOD_ZSTOD_API_CALL.value,
        ]
