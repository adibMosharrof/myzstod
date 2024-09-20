from model_loaders.base_model_loader import BaseModelLoader
from peft import PeftModel, PeftConfig, LoraConfig, TaskType


class LoraModelLoader(BaseModelLoader):

    def load(self):
        pass

    def get_modules_to_save(self) -> list[str]:
        if "gpt-j" in self.model_name:
            return ["lm_head", "wte"]
        if "t5" in self.model_name:
            return ["encoder.embed_tokens", "decoder.embed_tokens", "lm_head", "shared"]
        return ["lm_head", "embed_tokens"]

    def get_lora_config(self) -> LoraConfig:
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
            modules_to_save=self.get_modules_to_save(),
        )
