from pathlib import Path
from typing import Union
from model_loaders.lora_model_loader import LoraModelLoader
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, PeftModel, get_peft_model
import torch


class QuantizedLoraModelLoader(LoraModelLoader):

    # def old_load(self, model_path: Union[Path, str] = None, is_inference: bool = False):
    def load(self, model_path: Union[Path, str] = None, is_inference: bool = False):
        model_path = self._get_model_path(model_path)
        config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = {"": self.accelerator.process_index}
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = self.model_class.from_pretrained(
            self.model_name,
            quantization_config=config,
            device_map=device_map,
            torch_dtype=dtype,
        )
        self._resize_token_embeddings(model)
        config = self._get_lora_config()
        if is_inference:
            model = PeftModel.from_pretrained(model, model_path)
            return model
        model.enable_input_require_grads()
        model = prepare_model_for_kbit_training(model)
        if self.resume_checkpoint:
            model = PeftModel.from_pretrained(
                model, self.resume_checkpoint, config=config, is_trainable=True
            )
            return model
        model = get_peft_model(model, config)
        return model

    # def load(self, model_path: Union[Path, str] = None, is_inference: bool = False):
    def old_load(self, model_path: Union[Path, str] = None, is_inference: bool = False):
        model_path = self._get_model_path(model_path)
        config = BitsAndBytesConfig(load_in_8bit=True)
        device_map = {"": self.accelerator.process_index}
        model = self.model_class.from_pretrained(
            self.model_name,
            quantization_config=config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        if is_inference:
            model = PeftModel.from_pretrained(model, model_path)
            return model
        if self.resume_checkpoint_path:
            model = PeftModel.from_pretrained(
                model, self.resume_checkpoint_path, config=config, is_trainable=True
            )
            return model
        return model
