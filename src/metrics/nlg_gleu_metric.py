import uuid

from logger.inference_logger_dataclasses import ServiceCallInferenceLogData
from metrics.tod_metrics_base import TodMetricsBase
from transformers import AutoTokenizer
import evaluate
import pandas as pd
import torch
import numpy as np


class NlgGleuMetric(TodMetricsBase):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.metric = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))

    def _update(self, predictions: list[str], references: list[str]) -> None:
        refs = np.expand_dims(references, axis=1)
        self.metric.add_batch(predictions=predictions, references=refs)

    def _compute(self) -> float:
        try:
            res = self.metric.compute()["google_bleu"]
        except:
            res = 0.0
        return res

    def compute_row(self, pred: str, ref: str) -> None:
        try:
            res = self.metric.compute(predictions=[pred], references=[[ref]])[
                "google_bleu"
            ]
        except:
            res = 0.0
        return round(res, 4)

    def __str__(self):
        res = self._compute()
        return f"Response GLEU: {res:.4f}"
