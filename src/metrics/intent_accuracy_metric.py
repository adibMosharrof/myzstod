from metrics.tod_metrics_base import TodMetricsBase
from my_enums import SpecialPredictions, SpecialTokens
from predictions_logger import IntentsPredictionLogger
import torch
import utils
import evaluate
from torchmetrics.classification import BinaryAccuracy
import numpy as np


class IntentAccuracyMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        # self.metric = evaluate.load("accuracy")
        # self.metric = BinaryAccuracy()
        self.prediction_logger = IntentsPredictionLogger()
        self.add_state("pred_intents", [], dist_reduce_fx="cat")
        # self.add_state("target_intents", [], dist_reduce_fx="cat")

    def _update(self, predictions: list[str], references: list[str]) -> None:
        target_intents = []
        pred_intents = []
        for target, prediction in zip(references, predictions):
            t_intents = self._extract_section_from_text(
                target,
                SpecialTokens.begin_intent,
                SpecialTokens.end_intent,
                multiple_values=True,
                default_value=[],
            )
            if not len(t_intents):
                continue
            # for _ in t_intents:
            # self.target_intents.append(torch.create_tensor(1))
            # self.target_intents.append(utils.create_tensor(1))
            p = self._extract_section_from_text(
                prediction,
                SpecialTokens.begin_intent,
                SpecialTokens.end_intent,
                [SpecialPredictions.DUMMY],
                multiple_values=True,
                trim_spaces=True,
            )
            for t_intent in t_intents:
                if t_intent in p:
                    # self.pred_intents.append(utils.create_tensor(1))
                    self.pred_intents.append(utils.create_tensor(1))
                    # self._log_prediction(t_intent, t_intent, True)
                else:
                    # self.pred_intents.append(utils.create_tensor(0))
                    self.pred_intents.append(utils.create_tensor(0))
                    # self._log_prediction(p[0], t_intent, False)

        # self.metric.add_batch(predictions=pred_intents, references=target_intents)

    def _compute(self) -> float:
        # return self.metric(
        #     utils.create_tensor(self.pred_intents),
        #     utils.create_tensor(self.target_intents),
        # )
        preds = utils.create_tensor(self.pred_intents)
        # targets = (utils.create_tensor(self.target_intents),)
        return torch.mean(preds, dtype=torch.float)
        # try:
        #     acc = (
        #         np.array(self.pred_intents) == np.array(self.target_intents)
        #     ).sum() / len(self.target_intents)
        # except ZeroDivisionError:
        #     acc = 0
        # return acc
        # return self.metric.compute()["accuracy"]
        # return self.metric.compute(
        #     predictions=self.pred_intents, references=self.target_intents
        # )["accuracy"]

    def __str__(self) -> str:
        score = self.compute()
        return f"Intent Accuracy:{score*100:.2f}"
