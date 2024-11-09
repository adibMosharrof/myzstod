from my_enums import ContextType
from tod.context_formatter.context_formatter_base import ContextFormatterBase
from tod.context_formatter.nlg_context_formatter import NlgContextFormatter
from tod.context_formatter.simpletod_context_formatter import SimpleTodContextFormatter
from tod.context_formatter.zstod_context_formatter import ZsTodContextFormatter
from utilities.context_manager import ContextManager


class ContextFormatterFactory:
    @staticmethod
    def create_context_formatter(context_type: str) -> ContextFormatterBase:
        if ContextManager.is_simple_tod(context_type):
            return SimpleTodContextFormatter()
        if ContextManager.is_zstod(context_type):
            return ZsTodContextFormatter()
        if any(
            [
                ContextManager.is_nlg_strategy(context_type),
                ContextManager.is_bitod(context_type),
                ContextManager.is_ketod(context_type),
            ]
        ):
            return NlgContextFormatter()
        else:
            raise ValueError(f"Unknown context type: {context_type}")
