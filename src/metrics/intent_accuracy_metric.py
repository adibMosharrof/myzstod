from metrics.tod_metrics_base import TodMetricsBase
from predictions_logger import IntentsPredictionLogger
from simple_tod_dataclasses import SpecialPredictions, SpecialTokens

import evaluate


class IntentAccuracyMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("accuracy")
        self.prediction_logger = IntentsPredictionLogger()

    def _add_batch(self, predictions: list[str], references: list[str]) -> None:
        target_intents = []
        pred_intents = []
        for target, prediction in zip(references, predictions):

            t = self._extract_section_from_text(
                target, SpecialTokens.begin_intent, SpecialTokens.end_intent
            )
            if not t:
                continue
            target_intents.append(1)
            p = self._extract_section_from_text(
                prediction,
                SpecialTokens.begin_intent,
                SpecialTokens.end_intent,
                SpecialPredictions.DUMMY,
            )
            if t == p:
                pred_intents.append(1)
                self._log_prediction(p, t, True)
            else:
                self._add_wrong_pred(t)
                self._log_prediction(p, t, False)
                pred_intents.append(0)

        self.metric.add_batch(predictions=pred_intents, references=target_intents)

    def _compute(self) -> float:
        return self.metric.compute()["accuracy"]

    def __str__(self) -> str:
        score = self.compute()
        return f"Intent Accuracy: {score*100:.2f}"
