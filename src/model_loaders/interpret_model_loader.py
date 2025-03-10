from model_loaders.base_model_loader import BaseModelLoader
from my_models.gpt2_mask_layers import GPT2MaskLayersLmHeadModel
import utils
from my_models.gpt2 import GPT2LMHeadModel
from transformers import T5ForConditionalGeneration


class InterpretModelLoader(BaseModelLoader):
    def __init__(self, model_name, tokenizer, project_root, resume_checkpoint=None):
        super().__init__(model_name, tokenizer, project_root, resume_checkpoint)

    def load_for_inference(self, model_path=None, config=None):
        self.model_class = self.get_model_class_for_inference(self.model_name)
        return super().load_for_inference(model_path, config)

    def get_model_class_for_inference(self, model_name: str):
        if utils.is_t5_model(model_name):
            return T5ForConditionalGeneration
        # return GPT2LMHeadModel
        return GPT2MaskLayersLmHeadModel
