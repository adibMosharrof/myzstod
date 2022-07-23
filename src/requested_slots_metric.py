from abc import ABC
from simple_tod_dataclasses import SpecialTokens
from simpletod_metrics import SimpleTodMetricsBase


class RequestedSlotsMetric(SimpleTodMetricsBase):
    def __init__(self) -> None:
        self.tp, self.fp, self.fn = 0, 0, 0
        self.slot_appear_num, self.slot_correct_num = {}, {}
        self.false_slots = set()
        self.requested_slots_misclassification = {}

    def add_batch(self, predictions: list[str], references: list[str]) -> any:
        for ref, pred in zip(references, predictions):
            try:
                target_slots_text = self._extract_section_from_text(
                    ref,
                    SpecialTokens.begin_requested_slots,
                    SpecialTokens.end_requested_slots,
                )
                target_slots = target_slots_text.split(",")
            except ValueError:
                continue
            try:
                pred_slots_text = self._extract_section_from_text(
                    pred,
                    SpecialTokens.begin_requested_slots,
                    SpecialTokens.end_requested_slots,
                )
                pred_slots = pred_slots_text.split(",")
            except ValueError:
                pred_slots = []

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

    def compute(self, metric=None) -> float:
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10) * 100
        return f1
