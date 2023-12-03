import torch
from metrics.tod_metrics_base import TodMetricsBase
import re
import utils


class ApiCallMethodMetric(TodMetricsBase):
    def __init__(
        self,
    ):
        super().__init__()
        self.add_state("method_accuracies", [], dist_reduce_fx="cat")

    def _update(self, predictions: list[str], references: list[str]) -> None:
        for pred_str, ref_str in zip(predictions, references):
            pred = self._get_method_from_text(pred_str)
            ref = self._get_method_from_text(ref_str)
            if pred == ref:
                self.method_accuracies.append(utils.create_tensor(1))
            else:
                self.method_accuracies.append(utils.create_tensor(0))

    def _get_method_from_text(self, text: str) -> str:
        reg_exp = r"method='([^']+)'"
        try:
            match = re.search(reg_exp, text).group(1)
        except:
            match = ""
        return match

    def _compute(self):
        accs = utils.create_tensor(self.method_accuracies)
        return torch.mean(accs, dtype=torch.float)

    def __str__(self):
        res = self._compute()
        return f"Service Call Method Accuracy: {res*100:.2f}"
