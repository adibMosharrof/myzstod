import torch
from metrics.tod_metrics_base import TodMetricsBase
import re
import utils


class ApiCallInvokeMetric(TodMetricsBase):
    def __init__(self, invoke_text: str = "ApiCall", name=""):
        super().__init__()
        self.add_state("invoke_accuracies", [], dist_reduce_fx="cat")
        self.invoke_text = invoke_text
        self.name = name

    def _update(self, predictions: list[str], references: list[str]) -> None:
        for pred_str in predictions:
            if self.invoke_text in pred_str:
                self.invoke_accuracies.append(utils.create_tensor(1))
            else:
                self.invoke_accuracies.append(utils.create_tensor(0))

    def _compute(self):
        accs = utils.create_tensor(self.invoke_accuracies)
        return torch.mean(accs, dtype=torch.float)

    def compute_row(self, pred, ref) -> int:
        if self.invoke_text in pred:
            return 1
        return 0

    def __str__(self):
        res = self._compute()
        return f"{self.name}{self.invoke_text} Invoke Accuracy: {res*100:.2f}"
