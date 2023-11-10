import uuid
import evaluate
import numpy as np

from logger.inference_logger import InferenceLogger


class NlgMetricManager:
    def __init__(self, logger):
        self.google_bleu = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        self.logger = logger

    def compute_metrics(self, inf_logger: InferenceLogger):
        gleu_labels = np.expand_dims(inf_logger.concat_labels, axis=1)
        result = self.google_bleu.compute(
            predictions=inf_logger.concat_preds, references=gleu_labels
        )
        score_str = f"GLEU score: {result['google_bleu']:.4f}"
        self.logger.info(score_str)
        print(score_str)

    def compute_single_row(self, preds, labels):
        result = self.google_bleu.compute(predictions=[preds], references=[[labels]])
        return round(result["google_bleu"], 4)
