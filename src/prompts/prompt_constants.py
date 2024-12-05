from enum import Enum


class NlgPromptType(str, Enum):
    DEFAULT = "default"
    MULTI_DOMAIN = "multi_domain"
    CHATGPT = "chatgpt"
    CROSS = "cross"
    AUTO_TOD = "auto_tod"

    @classmethod
    def list(cls):
        return [c.value for c in cls]
