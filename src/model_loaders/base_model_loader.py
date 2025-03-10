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
import utils


class BaseModelLoader:

    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        project_root: Path,
        resume_checkpoint: str = None,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.project_root = project_root
        self.accelerator = Accelerator()
        self.model_class = self._get_model_class(self.model_name)

        self.resume_checkpoint = (
            str(project_root / resume_checkpoint) if resume_checkpoint else None
        )

    def load(
        self,
        model_path: Union[Path, str] = None,
        is_inference: bool = False,
        config=None,
    ) -> Union[AutoModelForCausalLM, T5ForConditionalGeneration]:
        model_path = self._get_model_path(model_path)

        model = self.model_class.from_pretrained(
            model_path or self.model_name, config=config
        )
        self._resize_token_embeddings(model)
        if model_path:
            model.to(self.accelerator.device)
        return model

    def load_for_inference(self, model_path: Union[Path, str] = None, config=None):
        return self.load(model_path, config=config)

    def _get_model_path(self, model_path: Union[Path, str] = None):
        return str(self.project_root / model_path) if model_path else None

    def _resize_token_embeddings(self, model):
        model.resize_token_embeddings(len(self.tokenizer))

    def _get_model_class(self, model_name: str):
        if utils.is_t5_model(model_name):
            return T5ForConditionalGeneration
        return AutoModelForCausalLM
