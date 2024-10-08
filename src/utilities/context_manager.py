from my_enums import ContextType


class ContextManager:
    @staticmethod
    def is_ketod(context_type: str) -> bool:
        return context_type in [
            ContextType.KETOD_API_CALL.value,
            ContextType.KETOD_GPT_API_CALL.value,
        ]

    @staticmethod
    def is_bitod(context_type: str) -> bool:
        return context_type in [ContextType.BITOD.value, ContextType.BITOD_GPT.value]

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
        ]
