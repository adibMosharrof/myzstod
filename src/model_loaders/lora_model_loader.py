from typing import Union
from model_loaders.base_model_loader import BaseModelLoader
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
from pathlib import Path
import torch


class LoraModelLoader(BaseModelLoader):

    def load(
        self, model_path: Union[Path, str] = None, is_inference: bool = False
    ) -> PeftModel:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_path = self._get_model_path(model_path)
        model = self.model_class.from_pretrained(
            model_path or self.model_name, load_in_8bit=False, torch_dtype=dtype
        )
        self._resize_token_embeddings(model)
        lora_config = self._get_lora_config()
        if self.resume_checkpoint:
            return PeftModel.from_pretrained(
                model, self.resume_checkpoint, config=lora_config, is_trainable=True
            )
        model = get_peft_model(model, lora_config)
        return model

    def load_for_inference(self, model_path: Union[Path, str] = None) -> PeftModel:
        device_map = {"": self.accelerator.device}
        model = self.model_class.from_pretrained(
            self.model_name, load_in_8bit=False, device_map=device_map
        )
        self._resize_token_embeddings(model)
        model = PeftModel.from_pretrained(model, model_path, device_map=device_map)
        model.eval()
        return model

    def _get_lora_config(self) -> LoraConfig:
        target_modules = None
        rank = 16
        if "t5" in self.model_name:
            task = TaskType.SEQ_2_SEQ_LM
            target_modules = ["q", "v"]
        else:
            task = TaskType.CAUSAL_LM
            target_modules = ["q_proj", "v_proj"]
        return LoraConfig(
            r=rank,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=task,
            base_model_name_or_path=self.model_name,
            target_modules=target_modules,
            modules_to_save=self._get_modules_to_save(),
        )

    def _get_modules_to_save(self) -> list[str]:
        if "gpt-j" in self.model_name:
            return ["lm_head", "wte"]
        if "t5" in self.model_name:
            return ["encoder.embed_tokens", "decoder.embed_tokens", "lm_head", "shared"]
        return ["lm_head", "embed_tokens"]
