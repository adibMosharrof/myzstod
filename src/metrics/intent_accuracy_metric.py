from metrics.tod_metrics_base import TodMetricsBase
from my_enums import SpecialPredictions, SpecialTokens
from predictions_logger import IntentsPredictionLogger

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

            t_intents = self._extract_section_from_text(
                target,
                SpecialTokens.begin_intent,
                SpecialTokens.end_intent,
                multiple_values=True,
            )
            if not len(t_intents):
                continue
            for _ in t_intents:
                target_intents.append(1)
            p = self._extract_section_from_text(
                prediction,
                SpecialTokens.begin_intent,
                SpecialTokens.end_intent,
                [SpecialPredictions.DUMMY],
                multiple_values=True,
            )
            for t_intent in t_intents:
                if t_intent in p:
                    pred_intents.append(1)
                    self._log_prediction(t_intent, t_intent, True)
                else:
                    pred_intents.append(0)
                    self._log_prediction(p[0], t_intent, False)

        self.metric.add_batch(predictions=pred_intents, references=target_intents)

    def _compute(self) -> float:
        return self.metric.compute()["accuracy"]

    def __str__(self) -> str:
        score = self.compute()
        return f"Intent Accuracy:{score*100:.2f}"
