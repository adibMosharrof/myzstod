from abc import ABC, abstractmethod


class BaseActivations(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def get_activations_and_confidences(
        self,
        interpret_text,
        generated_tokens,
        fc_vals,
        confidences,
        tokenizer,
        accelerator,
    ):
        raise NotImplementedError()
