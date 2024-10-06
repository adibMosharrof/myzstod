from my_models.gpt2_cross_attention import GPT2WithCrossAttention
from typing import Union
from pathlib import Path
from model_loaders.base_model_loader import BaseModelLoader


class CrossModelLoader(BaseModelLoader):
    def load(
        self, model_path: Union[Path, str] = None, is_inference: bool = False
    ) -> GPT2WithCrossAttention:
        model_path = self._get_model_path(model_path)

        model = GPT2WithCrossAttention.from_pretrained(model_path or self.model_name)
        self._resize_token_embeddings(model)
        if model_path:
            model.to(self.accelerator.device)
        return model
