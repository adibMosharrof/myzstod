# Joint GA, avg GA, Intent acc, Req Slot F1
"""
"dialog_id","turn_id","context","target"
"92_00000","1","<|begincontext|><|user|>I'm looking for some kid-friendly <Travel_category> attractions in <Travel_location>. Can you show me some?<|endcontext|>

","<|beginintent|>FindAttractions<|endintent|>

<|beginbelief|>Travel_category: Museum, Travel_goodForKids: True, Travel_location: Paris<|endbelief|>

<|beginaction|>OFFER Travel_attractionName, OFFER Travel_category<|endaction|>

<|beginresponse|>Sure. There is a <Travel_category> named the <Travel_attractionName> that meets your needs.<|endresponse|>

"
"""
import abc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import evaluate
import numpy as np
from sklearn.metrics import f1_score

from predictions_logger import (
    ActionGoalPredictionLogger,
    GoalPredictionLogger,
    IntentsPredictionLogger,
    PredictionsLoggerBase,
    RequestedSlotPredictionLogger,
)
from simple_tod_dataclasses import (
    GoalMetricConfigType,
    SimpleTodAction,
    SimpleTodBelief,
    SimpleTodConstants,
    SpecialPredictions,
    SpecialTokens,
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
                GoalPredictionLogger(step=GoalMetricConfigType.ACTION),
                SimpleTodAction,
            )
        elif step == GoalMetricConfigType.BELIEF:
            return GoalMetricConfig(
                SpecialTokens.begin_belief,
                SpecialTokens.end_belief,
                GoalMetricConfigType.BELIEF,
                GoalPredictionLogger(step=GoalMetricConfigType.BELIEF),
                SimpleTodBelief,
            )
        else:
            raise ValueError(f"Unknown step name: {step}")


class TodMetricsBase(ABC):
    """Base class for all TOD metrics."""

    def __init__(
        self,
        score: bool = 0.0,
        is_cached=False,
        prediction_logger: PredictionsLoggerBase = None,
    ):
        self.score = score
        self.is_cached = is_cached
        self.wrong_preds = {}
        self.prediction_logger = prediction_logger

    def _add_wrong_pred(self, key: any):
        if type(key) not in [int, str]:
            key = str(key)
        self.wrong_preds[key] = self.wrong_preds.get(key, 0) + 1

    def _log_prediction(
        self, pred: str = None, ref: str = None, is_correct: any = None
    ):
        self.prediction_logger.log(pred, ref, is_correct)

    def visualize(self, out_dir: Path) -> None:
        if not self.prediction_logger:
            return
        self.prediction_logger.visualize(out_dir)

    def _extract_section_from_text(
        self, text: str, start_token: str, end_token: str, default_value: any = None
    ) -> Optional[str]:
        try:
            idx1 = text.index(start_token)
            idx2 = text.index(end_token)
            res = text[idx1 + len(start_token) : idx2]
            return res
        except ValueError:
            return default_value

    def _extract_section_and_split_items_from_text(
        self,
        text: str,
        start_token: str,
        end_token: str,
        separator: str = SimpleTodConstants.ITEM_SEPARATOR,
        default_value: any = None,
    ) -> list[str]:
        section_txt = self._extract_section_from_text(
            text, start_token, end_token, default_value
        )
        if not section_txt:
            return []
        return section_txt.split(separator)

    def add_batch(self, predictions: list[str], references: list[str]) -> None:
        if not len(predictions):
            raise ValueError("You must provide at least one prediction.")
        if not len(references):
            raise ValueError("You must provide at least one reference.")
        if not len(predictions) == len(references):
            raise ValueError("Predictions and references must have the same length")
        self.is_cached = False
        return self._add_batch(predictions, references)

    @abc.abstractmethod
    def _add_batch(self, predictions: list[str], references: list[str]) -> None:
        pass

    def compute(self) -> float:
        if self.is_cached:
            return self.score
        self.score = self._compute()
        self.is_cached = True
        return self.score

    @abc.abstractmethod
    def _compute(self) -> float:
        pass


class MetricCollection:
    """Collects multiple metrics.
    Args:
        metrics: A dictionary of metrics.
    Example Usage:
        metrics = MetricCollection(
            {
                "goal_accuracy": GoalAccuracyMetric(),
                "intent_accuracy": IntentAccuracyMetric(),
                "requested_slots": RequestedSlotsMetric(),
            }
        )
        references = # list of whole target str
        predictions = # list of whole prediction str
        metrics.add_batch(predictions, references)
    """

    def __init__(self, metrics: dict[str, TodMetricsBase] = None):
        if metrics is None:
            raise ValueError("No metrics provided to MetricCollection")
        self.metrics = metrics

    def add_batch(self, predictions: list[str], references: list[str]) -> None:
        for m in self.metrics.values():
            m.add_batch(predictions, references)

    def compute(self) -> float:
        return [m.compute() for m in self.metrics.values()]

    def visualize(self, out_dir: Path) -> None:
        return [m.visualize(out_dir) for m in self.metrics.values()]

    def __str__(self):
        return "\n".join([str(m) for m in self.metrics.values()])


class IntentAccuracyMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__(prediction_logger=IntentsPredictionLogger())
        self.metric = evaluate.load("accuracy")

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


class RequestedSlotsMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__(prediction_logger=RequestedSlotPredictionLogger())
        self.all_preds = []
        self.all_refs = []

    def _add_batch(self, predictions: list[str], references: list[str]) -> any:
        for ref, pred in zip(references, predictions):

            target_slots = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )
            if not len(target_slots):
                continue
            pred_slots = self._extract_section_and_split_items_from_text(
                pred,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )

            if len(pred_slots) < len(target_slots):
                diff = len(target_slots) - len(pred_slots)
                pred_slots.extend([SpecialPredictions.DUMMY] * diff)

            for i, slot in enumerate(target_slots):
                if slot in pred_slots:
                    self.all_preds.append(slot)
                    self.all_refs.append(slot)
                    self._log_prediction(ref=slot, pred=slot, is_correct=True)
                else:
                    self.all_preds.append(pred_slots[i])
                    self.all_refs.append(slot)
                    self._log_prediction(ref=slot, pred=pred_slots[i], is_correct=False)

    def _compute(self) -> float:
        return f1_score(self.all_refs, self.all_preds, average="macro") * 100

    def __str__(self) -> str:
        score = self.compute()
        return f"Requested Slots Macro F1: {score:.2f}"


class BleuMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bleu")

    def _add_batch(self, predictions: list[str], references: list[str]) -> None:
        self.metric.add_batch(predictions=predictions, references=references)

    def _compute(self) -> float:
        return self.metric.compute()["bleu"]

    def __str__(self) -> str:
        score = self.compute()
        return f"BLEU: {score*100:.2f}"


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
                target_items = [t for t in target_items if t.value != ""]
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


class SuccessMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.all_success = []

    def _add_batch(self, turn_predictions: list[str], references: list[str]) -> None:
        for ref, pred in zip(references, turn_predictions):
            target_items = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )
            if not len(target_items):
                continue
            pred_items = self._extract_section_and_split_items_from_text(
                pred,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )
            batch_success = []
            for t in target_items:
                if t in pred_items:
                    batch_success.append(1)
                else:
                    self._add_wrong_pred(t)
                    batch_success.append(0)
            if not len(batch_success):
                continue
            self.all_success.append(np.mean(batch_success))

    def _compute(self) -> float:
        return np.mean(self.all_success)

    def __str__(self) -> str:
        avg_success = self.compute()
        return f"Success: {avg_success*100:.2f}"


class InformMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.all_inform = []

    def _add_batch(self, turn_predictions: list[str], references: list[str]) -> None:
        for ref, pred in zip(references, turn_predictions):
            target_items = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
            )
            pred_items = self._extract_section_and_split_items_from_text(
                pred,
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
            )
            target_actions = [SimpleTodAction.from_string(t) for t in target_items]
            pred_actions = [SimpleTodAction.from_string(p) for p in pred_items]
            batch_inform = []
            for t in target_actions:
                if not t.is_inform():
                    continue
                if t in pred_actions:
                    batch_inform.append(1)
                else:
                    self._add_wrong_pred(t)
                    batch_inform.append(0)
            if not len(batch_inform):
                continue
            self.all_inform.append(np.mean(batch_inform))

    def _compute(self) -> float:
        return np.mean(self.all_inform)

    def __str__(self) -> str:
        avg_inform = self.compute()
        return f"Inform: {avg_inform*100:.2f}"


class ResponseBleuMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("bleu")

    def _add_batch(self, predictions: list[str], references: list[str]) -> None:
        pred_responses_batch = []
        target_responses_batch = []
        for pred, ref in zip(predictions, references):
            target_response = self._extract_section_from_text(
                ref,
                SpecialTokens.begin_response,
                SpecialTokens.end_response,
            )
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
        return self.metric.compute()["bleu"]

    def __str__(self) -> str:
        score = self.compute()
        return f"Response BLEU: {score*100:.2f}"


class CombinedMetric(TodMetricsBase):
    def __init__(
        self,
        inform: InformMetric,
        success: SuccessMetric,
        response_bleu: ResponseBleuMetric,
    ) -> None:
        super().__init__()
        self.inform = inform
        self.success = success
        self.response_bleu = response_bleu

    def _add_batch(self, turn_predictions: list[str], references: list[str]) -> None:
        return

    def _compute(self) -> float:
        return (
            0.5 * (self.inform.compute() + self.success.compute())
            + self.response_bleu.compute()
        )

    def __str__(self) -> str:
        score = self.compute()
        return f"Combined: {score*100:.2f}"
