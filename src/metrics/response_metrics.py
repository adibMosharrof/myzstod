import numpy as np
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import ContextType, ResponseMetricType, SpecialTokens
import evaluate
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import BLEUScore
import torch
import uuid
import utils


class ResponseMetric(TodMetricsBase):
    def __init__(
        self,
        metric_name="bleu",
        metric_key_name=None,
        context_type=ContextType.SHORT_REPR.value,
    ) -> None:
        super().__init__()
        self.context_type = context_type
        self.metric_name = metric_name
        # self.metric = evaluate.load(metric_name, experiment_id=str(uuid.uuid4()))
        self.metric = (
            evaluate.load("rouge", experiment_id=str(uuid.uuid4()))
            # if metric_name == ResponseMetricType.ROUGE else BLEUScore()
            if metric_name == ResponseMetricType.ROUGE
            else evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
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
            if self.context_type == ContextType.NLG.value:
                target_response = ref
            else:
                target_response = self._extract_section_from_text(
                    ref,
                    SpecialTokens.begin_response,
                    SpecialTokens.end_response,
                )
            if not target_response:
                continue
            if self.context_type == ContextType.NLG.value:
                pred_response = pred
            else:
                pred_response = self._extract_section_from_text(
                    pred,
                    SpecialTokens.begin_response,
                    SpecialTokens.end_response,
                    "",
                )

            # pred_responses_batch.append(pred_response)
            # target_responses_batch.append(target_response)
            pred_responses_batch.append(pred_response)
            if self.metric_name == ResponseMetricType.ROUGE:
                target_responses_batch.append(target_response)
            elif self.metric_name == ResponseMetricType.BLEU:
                target_responses_batch.append([target_response])
        # if self.metric_name == ResponseMetricType.BLEU:
        if len(target_responses_batch) == 0:
            return
        self.metric.add_batch(
            predictions=pred_responses_batch, references=target_responses_batch
        )
        # self.pred_responses.append(pred_responses_batch)
        # self.target_responses.append(target_responses_batch)
        # self.metric.update(pred_response, target_response)
        # else:
        #     self.metric.update(pred_response, target_response)

    def _compute_old(self) -> float:
        # def _compute(self) -> float:
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

    def _compute(self) -> float:
        try:
            out = self.metric.compute()
            res = out[self.metric_key_name]
        except (ZeroDivisionError, ValueError):
            return utils.create_tensor(0.0, dtype=torch.float)

        if self.metric_name == ResponseMetricType.ROUGE:
            return torch.tensor(res.mid.fmeasure, dtype=torch.float)
        return torch.tensor(res, dtype=torch.float)

    def __str__(self) -> str:
        score = self.compute()
        return f"Response {self.metric_name.upper()}:{score*100:.2f}"
