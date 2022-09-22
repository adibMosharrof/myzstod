from dataclasses import dataclass
from typing import Union

import numpy as np
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import GoalMetricConfigType, SpecialTokens
from predictions_logger import (
    PredictionLoggerFactory,
    PredictionsLoggerBase,
    TodMetricsEnum,
)

from simple_tod_dataclasses import (
    SimpleTodAction,
    SimpleTodBelief,
)


@dataclass
class GoalMetricConfig:
    start_token: str
    end_token: str
    step_name: GoalMetricConfigType
    prediction_logger: PredictionsLoggerBase
    tod_class: Union[SimpleTodBelief, SimpleTodAction]


class GoalMetricConfigFactory:
    @staticmethod
    def create(step) -> GoalMetricConfig:
        if step == GoalMetricConfigType.ACTION:
            return GoalMetricConfig(
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
                GoalMetricConfigType.ACTION,
                PredictionLoggerFactory.create(TodMetricsEnum.ACTION),
                SimpleTodAction,
            )
        elif step == GoalMetricConfigType.BELIEF:
            return GoalMetricConfig(
                SpecialTokens.begin_belief,
                SpecialTokens.end_belief,
                GoalMetricConfigType.BELIEF,
                PredictionLoggerFactory.create(TodMetricsEnum.BELIEF),
                SimpleTodBelief,
            )
        else:
            raise ValueError(f"Unknown step name: {step}")


class GoalMetric(TodMetricsBase):
    """
    Computes avg and joint goal accuracy of belief or action.
    args:
        target_step_class: One of the following
            * SimpleTodAction: it will calculate action accuracy
            * SimpleTodBelief: it will calculate belief accuracy
    """

    def __init__(self, config: GoalMetricConfig) -> None:
        super().__init__()
        self.all_accuracies = []
        self.joint_accuracies = []
        self.wrong_preds = {}
        self.config = config
        self.prediction_logger = config.prediction_logger

    def _add_batch(self, turn_predictions: list[str], references: list[str]) -> None:
        for ref, pred in zip(references, turn_predictions):
            target_txt_items = self._extract_section_and_split_items_from_text(
                ref, self.config.start_token, self.config.end_token
            )
            if not len(target_txt_items):
                continue
            target_items = [
                self.config.tod_class.from_string(t) for t in target_txt_items
            ]
            if self.config.tod_class is SimpleTodBelief:
                target_items = [t for t in target_items if t.values != ""]
            pred_belief_txt_items = self._extract_section_and_split_items_from_text(
                pred, self.config.start_token, self.config.end_token
            )
            pred_beliefs = [
                self.config.tod_class.from_string(t) for t in pred_belief_txt_items
            ]

            turn_predictions = []
            any_wrong_preds = False
            for t in target_items:
                if t in pred_beliefs:
                    turn_predictions.append(1)
                    self._log_prediction(ref=t, is_correct=True)
                else:
                    turn_predictions.append(0)
                    self._log_prediction(ref=t, is_correct=False)
                    any_wrong_preds = True

            self.joint_accuracies.append(0 if any_wrong_preds else 1)
            self.all_accuracies.append(np.mean(turn_predictions))

    def _compute(self) -> float:
        return np.mean(self.all_accuracies), np.mean(self.joint_accuracies)

    def __str__(self) -> str:
        avg_ga, joint_ga = self.compute()
        return f"Average {self.config.step_name} Accuracy: {avg_ga*100:.2f}, Joint {self.config.step_name} Accuracy: {joint_ga*100:.2f}"
