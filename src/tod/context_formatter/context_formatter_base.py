from abc import ABC, abstractmethod

from tod.zs_tod_context import ZsTodContext


class ContextFormatterBase(ABC):
    @abstractmethod
    def to_str(self, context: ZsTodContext) -> str:
        pass
