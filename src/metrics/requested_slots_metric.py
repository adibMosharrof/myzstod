import random
import numpy as np
from sklearn.metrics import f1_score
import torch
from sgd_dstc8_data_model.dstc_dataclasses import DstcRequestedSlot
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import SpecialPredictions, SpecialTokens
from predictions_logger import PredictionLoggerFactory, TodMetricsEnum
from torchmetrics import F1Score
from collections import Counter
import utils


class RequestedSlotsMetric(TodMetricsBase):
    # def __init__(self) -> None:
    def __init_old__(self) -> None:
        super().__init__()
        # self.all_preds = []
        # self.all_refs = []
        self.prediction_logger = PredictionLoggerFactory.create(
            TodMetricsEnum.REQUESTED_SLOTS
        )
        self.add_state("all_refs", [], dist_reduce_fx="cat")
        self.add_state("all_preds", [], dist_reduce_fx="cat")
        # self.metrics = [F1Score(average="micro"), F1Score(average="macro")]

    def __init__(self) -> None:
        super().__init__()
        self.prediction_logger = PredictionLoggerFactory.create(
            TodMetricsEnum.REQUESTED_SLOTS
        )
        self.add_state("all_f1", [], dist_reduce_fx="cat")
        # self.add_state("all_preds", [], dist_reduce_fx="cat")
        # self.metrics = [F1Score(average="micro"), F1Score(average="macro")]

    def _update_old(self, predictions: list[str], references: list[str]) -> any:
        for ref, pred in zip(references, predictions):
            target_txt_items = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )
            target_slots = [DstcRequestedSlot.from_string(t) for t in target_txt_items]

            if not len(target_slots):
                continue
            pred_txt_items = self._extract_section_and_split_items_from_text(
                pred,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
                trim_spaces=True,
            )
            pred_slots = [DstcRequestedSlot.from_string(t) for t in pred_txt_items]

            if len(pred_slots) == 0 and len(target_slots) == 0:
                self.all_preds.append(SpecialPredictions.DUMMY)
                self.all_refs.append(SpecialPredictions.DUMMY)

            for i, slot in enumerate(target_slots):
                if slot in pred_slots:
                    self.all_preds.append(str(slot))
                    self.all_refs.append(str(slot))
                    self._log_prediction(ref=slot, pred=slot, is_correct=True)
                else:
                    if not len(pred_slots):
                        rand_pred = SpecialPredictions.DUMMY
                    elif len(pred_slots) < i:
                        rand_pred = str(pred_slots[i])
                    else:
                        rand_pred = str(random.choice(pred_slots))
                    self.all_preds.append(rand_pred)
                    self.all_refs.append(str(slot))
                    self._log_prediction(ref=slot, pred=rand_pred, is_correct=False)

    def _compute_old(self) -> float:
        return (
            f1_score(self.all_refs, self.all_preds, average="macro") * 100,
            f1_score(self.all_refs, self.all_preds, average="micro") * 100,
        )
        # return [m(self.all_preds, self.all_refs) for m in self.metrics]

    def __str_old__(self) -> str:
        macro_score, micro_score = self.compute()
        return f"Requested Slots Macro, Micro F1:{macro_score:.2f}, {micro_score:.2f}"

    def __str__(self) -> str:
        score = self.compute()
        return f"Requested Slots F1:{score:.2f}"

    def _update(self, predictions: list[str], references: list[str]) -> any:
        for ref, pred in zip(references, predictions):
            target_txt_items = self._extract_section_and_split_items_from_text(
                ref,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )
            target_slots = [
                DstcRequestedSlot.from_string(t).slot_name for t in target_txt_items
            ]

            pred_txt_items = self._extract_section_and_split_items_from_text(
                pred,
                SpecialTokens.begin_requested_slots,
                SpecialTokens.end_requested_slots,
            )
            pred_slots = [
                DstcRequestedSlot.from_string(t).slot_name for t in pred_txt_items
            ]
            ref_counter = Counter(target_slots)
            pred_counter = Counter(pred_slots)
            true = sum(ref_counter.values())
            positive = sum(pred_counter.values())
            true_positive = sum((ref_counter & pred_counter).values())
            precision = float(true_positive) / positive if positive else 1.0
            recall = float(true_positive) / true if true else 1.0
            if precision + recall > 0.0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:  # The F1-score is defined to be 0 if both precision and recall are 0.
                f1 = 0.0
            self.all_f1.append(utils.create_tensor(f1, dtype=torch.float))

    def _compute(self) -> float:
        f1_tensors = utils.create_tensor(self.all_f1, dtype=torch.float)
        return torch.mean(f1_tensors, dtype=torch.float) * 100
