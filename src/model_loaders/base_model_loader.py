from pathlib import Path
from typing import Union

from pathlib import Path
from typing import Union

from transformers import (
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)
from accelerate import Accelerator


class BaseModelLoader:

    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        project_root: Path,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.project_root = project_root
        self.accelerator = Accelerator()

    def load(
        self, model_path: Union[Path, str] = None
    ) -> Union[AutoModelForCausalLM, T5ForConditionalGeneration]:
        model_path = str(self.project_root / model_path) if model_path else None
        model_class = self.get_model_class(self.model_name)
        model = model_class.from_pretrained(model_path or self.model_name)
        self.resize_token_embeddings(model)
        if model_path:
            model = self.accelerator.prepare(model)
        return model

    def resize_token_embeddings(self, model):
        model.resize_token_embeddings(len(self.tokenizer))

    def get_model_class(self, model_name: str):
        config = AutoConfig.from_pretrained(model_name)
        if config.is_encoder_decoder:
            return T5ForConditionalGeneration
        return AutoModelForCausalLM
