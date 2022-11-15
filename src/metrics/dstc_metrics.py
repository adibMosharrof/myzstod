import numpy as np
from metrics.response_metrics import ResponseMetric
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import SpecialTokens
from predictions_logger import PredictionLoggerFactory, TodMetricsEnum
from simple_tod_dataclasses import (
    SimpleTodAction,
)
from dstc_dataclasses import DstcRequestedSlot


class SuccessMetric(TodMetricsBase):
    def __init__(self, slot_categories: dict[str, bool]) -> None:
        super().__init__()
        # self.all_success = []
        self.slot_categories = slot_categories
        self.prediction_logger = PredictionLoggerFactory.create(TodMetricsEnum.SUCCESS)
        self.add_state("all_success", [], dist_reduce_fx="cat")

    def _update(self, turn_predictions: list[str], references: list[str]) -> None:
        for ref, pred in zip(references, turn_predictions):
            requested_slots_txt = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
                multiple_values=True,
            )
            requested_slots = [
                DstcRequestedSlot.from_string(t) for t in requested_slots_txt
            ]
            if not len(requested_slots):
                continue
            target_actions_txt = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
            )
            target_actions = [
                SimpleTodAction.from_string(t, self.slot_categories)
                for t in target_actions_txt
            ]
            target_items = [act for act in target_actions if act in requested_slots]
            pred_items_txt = self._extract_section_and_split_items_from_text(
                pred,
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
            )
            pred_items = [
                SimpleTodAction.from_string(t, self.slot_categories)
                for t in pred_items_txt
            ]

            # batch_success = []
            for t in target_items:
                if t in pred_items:
                    self.all_success.append(1)
                    self._log_prediction(ref=t, is_correct=True)
                else:
                    self._add_wrong_pred(t)
                    self.all_success.append(0)
                    self._log_prediction(ref=t, is_correct=False)
            # if not len(batch_success):
            #     continue
            # self.all_success.append(np.mean(batch_success))

    def _compute(self) -> float:
        return np.mean(self.all_success)

    def __str__(self) -> str:
        avg_success = self.compute()
        return f"Success:{avg_success*100:.2f}"


class InformMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        # self.all_inform = []
        self.prediction_logger = PredictionLoggerFactory.create(TodMetricsEnum.INFORM)
        self.add_state("all_inform", [], dist_reduce_fx="cat")

    def _check(self, target: SimpleTodAction, preds: list[SimpleTodAction]) -> bool:
        for p in preds:
            if p.slot_name == target.slot_name:
                return True
        return False

    def _update(self, turn_predictions: list[str], references: list[str]) -> None:
        for ref, pred in zip(references, turn_predictions):
            target_items = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
            )
            if not len(target_items):
                continue
            pred_items = self._extract_section_and_split_items_from_text(
                pred,
                SpecialTokens.begin_action,
                SpecialTokens.end_action,
            )
            target_actions = [SimpleTodAction.from_string(t) for t in target_items]
            pred_actions = [SimpleTodAction.from_string(p) for p in pred_items]
            # batch_inform = []
            for t in target_actions:
                if not t.is_inform():
                    continue
                # if t in pred_actions:
                if self._check(t, pred_actions):
                    self.all_inform.append(1)
                    self._log_prediction(ref=t, is_correct=True)
                else:
                    self.all_inform.append(0)
                    self._log_prediction(ref=t, is_correct=False)
            # if not len(batch_inform):
            #     continue
            # self.all_inform.append(np.mean(batch_inform))

    def _compute(self) -> float:
        return np.mean(self.all_inform)

    def __str__(self) -> str:
        avg_inform = self.compute()
        return f"Inform:{avg_inform*100:.2f}"


class CombinedMetric(TodMetricsBase):
    def __init__(
        self,
        inform: InformMetric,
        success: SuccessMetric,
        response_bleu: ResponseMetric,
    ) -> None:
        super().__init__()
        self.inform = inform
        self.success = success
        self.response_bleu = response_bleu

    def _update(self, turn_predictions: list[str], references: list[str]) -> None:
        return

    def _compute(self) -> float:
        return (
            0.5 * (self.inform.compute() + self.success.compute())
            + self.response_bleu.compute()
        )

    def __str__(self) -> str:
        score = self.compute()
        return f"Combined:{score*100:.2f}"
