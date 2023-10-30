import uuid
import evaluate
import numpy as np


class NlgMetricManager:
    def __init__(self, logger):
        self.google_bleu = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        self.logger = logger

    def compute_metrics(self, labels, preds):
        gleu_labels = np.expand_dims(labels.concat_labels, axis=1)
        result = self.google_bleu.compute(predictions=preds, references=gleu_labels)
        score_str = f"GLEU score: {result['google_bleu']}"
        self.logger.info(score_str)
        print(score_str)
