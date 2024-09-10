from abc import ABC, abstractmethod


class SchemaEncoder(ABC):
    """
    Abstract base class for schema encoders.
    """

    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def encode(self, schemas: list):
        raise NotImplementedError()
