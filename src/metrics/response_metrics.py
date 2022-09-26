from metrics.tod_metrics_base import TodMetricsBase
from my_enums import SpecialTokens
import evaluate


class ResponseMetric(TodMetricsBase):
    def __init__(self, metric_name="bleu", metric_key_name=None) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.metric = evaluate.load(metric_name)
        self.metric_key_name = metric_key_name or metric_name

    def _add_batch(self, predictions: list[str], references: list[str]) -> None:
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
            pred_responses_batch.append(pred_response)
            target_responses_batch.append(target_response)
        self.metric.add_batch(
            predictions=pred_responses_batch, references=target_responses_batch
        )

    def _compute(self) -> float:
        try:
            res = self.metric.compute()[self.metric_key_name]
        except ZeroDivisionError:
            res = 0.0
        if self.metric_name == "rouge":
            return res.mid.fmeasure
        return res

    def __str__(self) -> str:
        score = self.compute()
        return f"Response {self.metric_name.upper()}: {score*100:.2f}"
