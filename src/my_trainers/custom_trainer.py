from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.utils import WEIGHTS_NAME
import os


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, model_loader=None, is_quantized_model=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_loader = model_loader
        self.is_quantized_model = is_quantized_model

    def _load_best_model(self):
        if not self.is_quantized_model:
            return super()._load_best_model()
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        return self.model_loader.load_for_inference(best_model_path)
