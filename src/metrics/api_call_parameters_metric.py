import torch
from metrics.tod_metrics_base import TodMetricsBase
import re
import utils


class ApiCallParametersMetric(TodMetricsBase):
    def __init__(
        self,
    ):
        super().__init__()
        self.add_state("param_preds", [], dist_reduce_fx="cat")
        self.add_state("value_preds", [], dist_reduce_fx="cat")

    def _update(self, predictions: list[str], references: list[str]) -> None:
        for pred, ref in zip(predictions, references):
            ref_params = self._get_parameters_from_text(ref)
            pred_params = self._get_parameters_from_text(pred)
            for k, v in ref_params.items():
                if k in pred_params.keys():
                    self.param_preds.append(utils.create_tensor(1))
                    fuzz_score = utils.fuzzy_string_match(pred_params[k], v)
                    self.value_preds.append(utils.create_tensor(fuzz_score))
                else:
                    self.param_preds.append(utils.create_tensor(0))
                    self.value_preds.append(utils.create_tensor(0))

    def _get_parameters_from_text(self, text: str) -> str:
        reg_exp = r"(\w+)': '([^']+)'"
        try:
            matches = re.findall(reg_exp, text)
            out = dict(matches)
        except:
            out = {}
        return out

    def _compute(self) -> tuple[float, float]:
        params_tensor = utils.create_tensor(self.param_preds)
        values_tensor = utils.create_tensor(self.value_preds)
        param_acc = torch.mean(params_tensor, dtype=torch.float)
        value_acc = torch.mean(values_tensor, dtype=torch.float)
        return param_acc, value_acc

    def __str__(self) -> str:
        params, values = self._compute()
        return f"Api Call Parameters Accuracy: {params*100:.2f}, Values Accuracy: {values*100:.2f}"
