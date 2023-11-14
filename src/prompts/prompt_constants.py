from enum import Enum


class NlgPromptType(str, Enum):
    DEFAULT = "default"
    MULTI_DOMAIN = "multi_domain"

    @classmethod
    def list(cls):
        return [c.value for c in cls]
