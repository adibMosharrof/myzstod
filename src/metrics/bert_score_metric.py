import uuid
from logger.inference_logger_dataclasses import (
    BertScoreData,
)
from metrics.tod_metrics_base import TodMetricsBase
from transformers import AutoTokenizer
import evaluate
import numpy as np


class BertScoreMetric(TodMetricsBase):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        bert_score_model="microsoft/mpnet-base",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.metric = evaluate.load("bertscore", experiment_id=str(uuid.uuid4()))
        self.row_metric = evaluate.load("bertscore", experiment_id=str(uuid.uuid4()))
        self.bert_score_model = bert_score_model

    def _update(self, predictions: list[str], references: list[str]) -> None:
        self.metric.add_batch(predictions=predictions, references=references)

    def _compute(self) -> BertScoreData:
        try:
            score = self.metric.compute(model_type=self.bert_score_model)
            res = BertScoreData(
                precision=np.mean(score["precision"]),
                recall=np.mean(score["recall"]),
                f1=np.mean(score["f1"]),
            )
        except ValueError:
            res = BertScoreData(precision=0.0, recall=0.0, f1=0.0)
        return res

    def compute_row(self, pred: str, ref: str) -> BertScoreData:
        score = self.row_metric.compute(
            predictions=[pred], references=[ref], model_type=self.bert_score_model
        )
        res = BertScoreData(
            precision=np.mean(score["precision"]),
            recall=np.mean(score["recall"]),
            f1=np.mean(score["f1"]),
        )
        return round(res.f1, 4)

    def __str__(self):
        res = self._compute()
        return str(res)
