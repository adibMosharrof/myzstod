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
from abc import ABC
import abc
from typing import Optional, Tuple, Union

from simple_tod_dataclasses import (
    SimpleTodAction,
    SimpleTodBelief,
    SimpleTodConstants,
    SpecialTokens,
)
import evaluate
import numpy as np


class TodMetricsBase(ABC):
    """Base class for all TOD metrics."""

    def is_value_same(self, a: str, b: str) -> int:
        return int(a == b)

    def _extract_section_from_text(
        self, text: str, start_token: str, end_token: str, default_value: any = ""
    ) -> Optional[str]:
        try:
            idx1 = text.index(start_token)
            idx2 = text.index(end_token)
            res = text[idx1 + len(start_token) : idx2]
            return res
        except ValueError:
            return default_value

    @abc.abstractmethod
    def add_batch(self, predictions: list[str], references: list[str]) -> None:
        pass

    @abc.abstractmethod
    def compute(self, metric=None) -> float:
        pass


class MetricCollection(TodMetricsBase):
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
        if not len(predictions) == len(references):
            raise ValueError("Predictions and references must have the same length")
        for m in self.metrics.values():
            m.add_batch(predictions, references)

    def compute(self) -> float:
        return [m.compute() for m in self.metrics.values()]

    def __str__(self):
        return "\n".join([str(m) for m in self.metrics.values()])


class IntentAccuracyMetric(TodMetricsBase):
    def __init__(self) -> None:
        self.metric = evaluate.load("accuracy")

    def add_batch(self, predictions: list[str], references: list[str]) -> None:
        target_intents = []
        pred_intents = []
        for target, prediction in zip(references, predictions):

            t = self._extract_section_from_text(
                target, SpecialTokens.begin_intent, SpecialTokens.end_intent, None
            )
            if not t:
                continue
            target_intents.append(1)
            p = self._extract_section_from_text(
                prediction, SpecialTokens.begin_intent, SpecialTokens.end_intent
            )
            pred_intents.append(self.is_value_same(t, p))

        self.metric.add_batch(predictions=pred_intents, references=target_intents)

    def compute(self) -> float:
        return self.metric.compute()

    def __str__(self) -> str:
        score = self.compute()
        return f"Intent Accuracy: {score['accuracy']*100:.2f}"


class RequestedSlotsMetric(TodMetricsBase):
    def __init__(self) -> None:
        self.tp, self.fp, self.fn = 0, 0, 0
        self.slot_appear_num, self.slot_correct_num = {}, {}
        self.false_slots = set()
        self.requested_slots_misclassification = {}

    def add_batch(self, predictions: list[str], references: list[str]) -> any:
        for ref, pred in zip(references, predictions):

            target_slots_text = self._extract_section_from_text(
                ref,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
                None,
            )
            if not target_slots_text:
                continue
            target_slots = target_slots_text.split(SimpleTodConstants.ITEM_SEPARATOR)
            pred_slots_text = self._extract_section_from_text(
                pred,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
                [],
            )
            pred_slots = pred_slots_text.split(SimpleTodConstants.ITEM_SEPARATOR)

            for slot in pred_slots:
                if slot in target_slots:
                    val = self.slot_correct_num.get(slot, 0)
                    self.slot_correct_num[slot] = val + 1
                    self.tp += 1
                else:
                    self.fp += 1
                    self.false_slots.add(slot)
            for slot in target_slots:
                val = self.slot_appear_num.get(slot, 0)
                self.slot_appear_num[slot] = val + 1
                if slot not in pred_slots:
                    self.fn += 1
                    self.false_slots.add(slot)

    def compute(self) -> float:
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10) * 100
        return f1

    def __str__(self) -> str:
        score = self.compute()
        return f"Requested Slots Macro F1: {score:.2f}"


class BleuMetric(TodMetricsBase):
    def __init__(self) -> None:
        self.metric = evaluate.load("bleu")

    def add_batch(self, predictions: list[str], references: list[str]) -> None:
        self.metric.add_batch(predictions=predictions, references=references)

    def compute(self) -> float:
        return self.metric.compute()

    def __str__(self) -> str:
        score = self.compute()
        return f"BLEU: {score['bleu']*100:.2f}"


class GoalMetric(TodMetricsBase):
    """
    Computes avg and joint goal accuracy of belief or action.
    args:
        target_step_class: One of the following
            * SimpleTodAction: it will calculate action accuracy
            * SimpleTodBelief: it will calculate belief accuracy
    """

    def __init__(
        self, target_step_class: Union[SimpleTodAction, SimpleTodBelief]
    ) -> None:
        self.avg_accuracies = []
        self.joint_accuracies = []
        self.wrong_preds = {}
        self.target_step_class = target_step_class
        if target_step_class is SimpleTodBelief:
            self.start_token = SpecialTokens.begin_belief
            self.end_token = SpecialTokens.end_belief
            self.step_name = "Goal"
        elif target_step_class is SimpleTodAction:
            self.start_token = SpecialTokens.begin_action
            self.end_token = SpecialTokens.end_action
            self.step_name = "Action"
        else:
            raise ValueError(
                f"Provided target_step_class:{target_step_class}, but it must be SimpleTodBelief or SimpleTodAction"
            )

    def add_batch(self, turn_predictions: list[str], references: list[str]) -> None:
        for ref, pred in zip(references, turn_predictions):
            target_txt = self._extract_section_from_text(
                ref, self.start_token, self.end_token, None
            )
            if not target_txt:
                continue
            target_items = [
                self.target_step_class.from_string(t)
                for t in target_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
            ]

            pred_belief_txt = self._extract_section_from_text(
                pred, self.start_token, self.end_token
            )
            pred_beliefs = [
                self.target_step_class.from_string(t)
                for t in pred_belief_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
            ]

            turn_predictions = []
            any_wrong_preds = False
            for t in target_items:
                if t in pred_beliefs:
                    turn_predictions.append(1)
                else:
                    turn_predictions.append(0)
                    self.wrong_preds[t] = self.wrong_preds.get(t, 0) + 1
                    any_wrong_preds = True

            self.joint_accuracies.append(0 if any_wrong_preds else 1)
            self.avg_accuracies.append(np.mean(turn_predictions))

    def compute(self) -> float:
        return np.mean(self.avg_accuracies), np.mean(self.joint_accuracies)

    def __str__(self) -> str:
        avg_ga, joint_ga = self.compute()
        return f"Average {self.step_name} Accuracy: {avg_ga*100:.2f}, Joint {self.step_name} Accuracy: {joint_ga*100:.2f}"
