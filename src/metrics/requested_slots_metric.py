import random
from sklearn.metrics import f1_score
from dstc_dataclasses import DstcRequestedSlot
from metrics.tod_metrics_base import TodMetricsBase
from my_enums import SpecialPredictions, SpecialTokens
from predictions_logger import PredictionLoggerFactory, TodMetricsEnum
from torchmetrics import F1Score


class RequestedSlotsMetric(TodMetricsBase):
    def __init__(self) -> None:
        super().__init__()
        # self.all_preds = []
        # self.all_refs = []
        self.prediction_logger = PredictionLoggerFactory.create(
            TodMetricsEnum.REQUESTED_SLOTS
        )
        self.add_state("all_refs", [], dist_reduce_fx="cat")
        self.add_state("all_preds", [], dist_reduce_fx="cat")
        # self.metrics = [F1Score(average="micro"), F1Score(average="macro")]

    def _update(self, predictions: list[str], references: list[str]) -> any:
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
            )
            pred_slots = [DstcRequestedSlot.from_string(t) for t in pred_txt_items]

            # if len(pred_slots) < len(target_slots):
            #     diff = len(target_slots) - len(pred_slots)
            #     pred_slots
            #     pred_slots.extend(
            #         [
            #             DstcRequestedSlot(
            #                 SpecialPredictions.DUMMY, SpecialPredictions.DUMMY
            #             )
            #             for _ in range(diff)
            #         ]
            #     )
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

    def _compute(self) -> float:
        return (
            f1_score(self.all_refs, self.all_preds, average="macro") * 100,
            f1_score(self.all_refs, self.all_preds, average="micro") * 100,
        )
        # return [m(self.all_preds, self.all_refs) for m in self.metrics]

    def __str__(self) -> str:
        macro_score, micro_score = self.compute()
        return f"Requested Slots Macro, Micro F1:{macro_score:.2f}, {micro_score:.2f}"
