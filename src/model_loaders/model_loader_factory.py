from model_loaders.cross_model_loader import CrossModelLoader
from transformers import AutoTokenizer

from model_loaders.base_model_loader import BaseModelLoader
from model_loaders.interpret_model_loader import InterpretModelLoader
from model_loaders.lora_model_loader import LoraModelLoader
from model_loaders.qlora_4bit_model_loader import Qlora4bitModelLoader
from model_loaders.quantized_lora_model_loader import QuantizedLoraModelLoader


class ModelLoaderFactory:

    @staticmethod
    def get_loader(cfg, tokenizer: AutoTokenizer) -> BaseModelLoader:
        params = {
            "model_name": cfg.model_type.model_name,
            "tokenizer": tokenizer,
            "project_root": cfg.project_root,
            "resume_checkpoint": cfg.resume_checkpoint,
        }

        for key, value in params.items():
            if value is None:
                raise ValueError(f"The parameter '{key}' cannot be None.")

        if cfg.model_type.context_type in ["gpt_cross"]:
            return CrossModelLoader(**params)
        if cfg.get("is_interpret", None):
            return InterpretModelLoader(**params)
        if not cfg.model_type.quantization:
            return BaseModelLoader(**params)
        if cfg.model_type.quantization_dtype == 16:
            return LoraModelLoader(**params)
        if cfg.model_type.quantization_dtype == 8:
            return QuantizedLoraModelLoader(**params)
        if cfg.model_type.quantization_dtype == 4:
            return Qlora4bitModelLoader(**params)
