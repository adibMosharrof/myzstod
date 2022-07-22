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
from collections import defaultdict
from itertools import zip_longest
import random
import re
from typing import Optional, Tuple

from simple_tod_dataclasses import SpecialTokens


class SimpleTodMetrics:
    def __init__(self) -> None:
        self.requested_slots = defaultdict(lambda: len(self.requested_slots))

    def is_value_same(self, a: str, b: str) -> int:
        return int(a == b)

    def _extract_section_from_text(
        self, text: str, start_token: str, end_token: str
    ) -> str:
        idx1 = text.index(start_token)
        idx2 = text.index(end_token)
        res = text[idx1 + len(start_token) : idx2]
        return res

    def get_f1_requested_slots_f1_score(self, tp: int, fp: int, fn: int) -> float:
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10) * 100
        return f1

    def get_requested_slots_metric_values(
        self,
        batch_preds_text: list[str],
        batch_targets_text: list[str],
        tp: int,
        fp: int,
        fn: int,
        slot_appear_num: dict,
        slot_correct_num: dict,
        false_slots: list[str],
    ):
        for target, prediction in zip(batch_targets_text, batch_preds_text):
            try:
                target_slots_text = self._extract_section_from_text(
                    target,
                    SpecialTokens.begin_requested_slots,
                    SpecialTokens.end_requested_slots,
                )
                target_slots = target_slots_text.split(",")
            except ValueError:
                continue
            try:
                pred_slots_text = self._extract_section_from_text(
                    prediction,
                    SpecialTokens.begin_requested_slots,
                    SpecialTokens.end_requested_slots,
                )
                pred_slots = pred_slots_text.split(",")
            except ValueError:
                pred_slots = []

            for slot in pred_slots:
                if slot in target_slots:
                    val = slot_correct_num.get(slot, 0)
                    slot_correct_num[slot] = val + 1
                    tp += 1
                else:
                    fp += 1
                    false_slots.append(slot)
            for slot in target_slots:
                val = slot_appear_num.get(slot, 0)
                slot_appear_num[slot] = val + 1
                if slot not in pred_slots:
                    fn += 1
                    false_slots.append(slot)
        return tp, fp, fn, slot_appear_num, slot_correct_num, list(set(false_slots))

    def get_intent_pred_ref(self, preds, targets) -> Tuple[list[int], list[int]]:
        target_intents = []
        pred_intents = []
        for target, prediction in zip(targets, preds):
            try:
                t = self._extract_section_from_text(
                    target, SpecialTokens.begin_intent, SpecialTokens.end_intent
                )
                target_intents.append(1)
            except ValueError:
                continue
            try:
                p = self._extract_section_from_text(
                    prediction, SpecialTokens.begin_intent, SpecialTokens.end_intent
                )
            except ValueError:
                p = ""
            finally:
                pred_intents.append(self.is_value_same(t, p))

        return pred_intents, target_intents


if __name__ == "__main__":
    stm = SimpleTodMetrics()
    pred = """<|beginintent|>FindAttractions<|endintent|>

<|beginbelief|>Travel_category: Museum, Travel_goodForKids: True, Travel_location: Paris<|endbelief|>

<|beginaction|>OFFER Travel_attractionName, OFFER Travel_category<|endaction|>

<|beginresponse|>Sure. There is a <Travel_category> named the <Travel_attractionName> that meets your needs.<|endresponse|>

"""

    acc = stm.get_intent_accuracy(pred, pred)
    print(acc)
