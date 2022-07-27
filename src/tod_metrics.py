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

    def __init__(self, score: bool = 0.0, is_cached=False):
        self.score = score
        self.is_cached = is_cached
        self.wrong_preds = {}

    def _add_wrong_pred(self, key: any):
        self.wrong_preds[key] = self.wrong_preds.get(key, 0) + 1

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

    def __str__(self):
        return "\n".join([str(m) for m in self.metrics.values()])


class IntentAccuracyMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.metric = evaluate.load("accuracy")

    def _add_batch(self, predictions: list[str], references: list[str]) -> None:
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
            if t == p:
                pred_intents.append(1)
            else:
                self._add_wrong_pred(t)
                pred_intents.append(0)

        self.metric.add_batch(predictions=pred_intents, references=target_intents)

    def _compute(self) -> float:
        return self.metric.compute()["accuracy"]

    def __str__(self) -> str:
        score = self.compute()
        return f"Intent Accuracy: {score*100:.2f}"


class RequestedSlotsMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.tp, self.fp, self.fn = 0, 0, 0
        self.slot_appear_num, self.slot_correct_num = {}, {}
        self.false_slots = set()
        self.requested_slots_misclassification = {}

    def _add_batch(self, predictions: list[str], references: list[str]) -> any:
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

    def _compute(self) -> float:
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10) * 100
        return f1

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

    def __init__(
        self, target_step_class: Union[SimpleTodAction, SimpleTodBelief]
    ) -> None:
        super().__init__()
        self.all_accuracies = []
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

    def _add_batch(self, turn_predictions: list[str], references: list[str]) -> None:
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
                    self._add_wrong_pred(t)
                    any_wrong_preds = True

            self.joint_accuracies.append(0 if any_wrong_preds else 1)
            self.all_accuracies.append(np.mean(turn_predictions))

    def _compute(self) -> float:
        return np.mean(self.all_accuracies), np.mean(self.joint_accuracies)

    def __str__(self) -> str:
        avg_ga, joint_ga = self.compute()
        return f"Average {self.step_name} Accuracy: {avg_ga*100:.2f}, Joint {self.step_name} Accuracy: {joint_ga*100:.2f}"


class SuccessMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        self.all_success = []

    def _add_batch(self, turn_predictions: list[str], references: list[str]) -> None:
        for ref, pred in zip(references, turn_predictions):
            target_txt = self._extract_section_from_text(
                ref,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
                None,
            )
            if not target_txt:
                continue
            target_items = target_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
            pred_txt = self._extract_section_from_text(
                pred,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
                [],
            )
            pred_items = pred_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
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
            target_txt = self._extract_section_from_text(
                ref,
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
                None,
            )
            pred_txt = self._extract_section_from_text(
                pred,
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
                [],
            )
            pred_items = pred_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
            target_items = target_txt.split(SimpleTodConstants.ITEM_SEPARATOR)
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
                None,
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
