import torch
from metrics.tod_metrics_base import TodMetricsBase
from utilities import text_utilities
from utilities import tensor_utilities
import numpy as np


class InterpretParamsMetric(TodMetricsBase):
    def __init__(self, target_param: str = ""):
        super().__init__()
        self.add_state("target_param_acc", [], dist_reduce_fx="cat")
        self.add_state("other_param_acc", [], dist_reduce_fx="cat")
        self.target_param = target_param

    def _update(self, predictions: list[str], references: list[str]) -> None:
        for pred, ref in zip(predictions, references):
            pred_params = text_utilities.get_parameters_from_text(pred)
            ref_params = text_utilities.get_parameters_from_text(ref)
            target_pred = self.get_target_param(pred_params)
            target_ref = self.get_target_param(ref_params)

            self.target_param_acc.append(self.get_acc(target_pred, target_ref))
            other_preds = [param for param in pred_params if param != target_pred]
            other_refs = [param for param in ref_params if param != target_ref]
            [
                self.other_param_acc.append(self.get_acc(other_pred, other_ref))
                for other_pred, other_ref in zip(other_preds, other_refs)
            ]

    def get_acc(self, pred, ref):
        return tensor_utilities.create_tensor(int(pred == ref), device=self.device)

    def _compute(self) -> tuple[float, float, float]:
        primary_acc = np.mean(self.target_param_acc)
        other_acc = np.mean(self.other_param_acc)
        return primary_acc, other_acc

    def __str__(self) -> str:
        primary_acc, other_acc = self._compute()
        return f"Primary Param Accuracy: {primary_acc*100:.2f}, Other Param Accuracy: {other_acc*100:.2f}"

    def interpret_repr(self):
        primary_acc, other_acc = self._compute()
        return [
            {"name": "Primary Param Accuracy", "value": primary_acc},
            {"name": "Other Param Accuracy", "value": other_acc},
        ]

    def get_target_param(self, params: list[str]):
        for param in params:
            if self.target_param in param:
                return param
        return None
