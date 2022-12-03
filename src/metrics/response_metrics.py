import numpy as np
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import ResponseMetricType, SpecialTokens
import evaluate
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import BLEUScore
import torch
import uuid


class ResponseMetric(TodMetricsBase):
    def __init__(self, metric_name="bleu", metric_key_name=None) -> None:
        super().__init__()
        self.metric_name = metric_name
        # self.metric = evaluate.load(metric_name, experiment_id=str(uuid.uuid4()))
        self.metric = (
            # ROUGEScore() if metric_name == ResponseMetricType.ROUGE else BLEUScore()
            ROUGEScore()
            if metric_name == ResponseMetricType.ROUGE
            else evaluate.load("google_bleu")
            # else BLEU()
            # else BLEUScore()
        )
        self.metric_key_name = metric_key_name or metric_name
        self.add_state("pred_responses", [], dist_reduce_fx="cat")
        self.add_state("target_responses", [], dist_reduce_fx="cat")

    def _update(self, predictions: list[str], references: list[str]) -> None:
        pred_responses_batch = []
        target_responses_batch = []
        for pred, ref in zip(predictions, references):
            target_response = self._extract_section_from_text(
                ref,
                SpecialTokens.begin_response,
                SpecialTokens.end_response,
            )
            if not target_response:
                continue
            pred_response = self._extract_section_from_text(
                pred,
                SpecialTokens.begin_response,
                SpecialTokens.end_response,
                "",
            )

            # pred_responses_batch.append(pred_response)
            # target_responses_batch.append(target_response)
            pred_responses_batch.append(pred_response)
            target_responses_batch.append([target_response])
        if self.metric_name == ResponseMetricType.BLEU:
            self.metric.add_batch(
                predictions=pred_responses_batch, references=target_responses_batch
            )
            # self.pred_responses.append(pred_responses_batch)
            # self.target_responses.append(target_responses_batch)
            # self.metric.update(pred_response, target_response)
        else:
            self.metric.update(pred_response, target_response)

    # def _compute_old(self) -> float:
    def _compute(self) -> float:
        try:
            res = self.metric.compute(
                # predictions=self.pred_responses, references=self.target_responses
            )[self.metric_key_name]
        except ZeroDivisionError:
            res = 0.0
        return res
        out = self.metric.compute()
        if self.metric_name == ResponseMetricType.ROUGE:
            return out["rouge2_fmeasure"]
        return out[self.metric_key_name]
        # return out

    def _compute_new(self):
        if self.metric_name == ResponseMetricType.ROUGE:
            out = self.metric.compute()
            return out[self.metric_key_name]
        preds = np.concatenate(self.pred_responses, axis=0).tolist()
        out = self.metric.corpus_score(preds, self.target_responses)
        return out.score

    def __str__(self) -> str:
        score = self.compute()
        return f"Response {self.metric_name.upper()}:{score*100:.2f}"
