import torch
from metrics.tod_metrics_base import TodMetricsBase
import re
import utils
import numpy as np


class ApiCallParametersMetric(TodMetricsBase):
    def __init__(self, name: str = ""):
        super().__init__()
        self.add_state("param_preds", [], dist_reduce_fx="cat")
        self.add_state("value_preds", [], dist_reduce_fx="cat")
        # self.add_state("only_value_preds", [], dist_reduce_fx="cat")
        self.name = name

    def _update(self, predictions: list[str], references: list[str]) -> None:
        for pred, ref in zip(predictions, references):
            ref_params = self._get_parameters_from_text(ref)
            pred_params = self._get_parameters_from_text(pred)
            for k, v in ref_params.items():
                # self.only_value_preds.append(
                #     utils.create_tensor(self.get_only_value_score(pred_params, v))
                # )
                if k in pred_params.keys():
                    fuzz_score = utils.fuzzy_string_match(pred_params[k], v)
                    self.param_preds.append(utils.create_tensor(1))
                    self.value_preds.append(utils.create_tensor(fuzz_score))
                else:
                    self.param_preds.append(utils.create_tensor(0))
                    self.value_preds.append(utils.create_tensor(0))

    def _get_parameters_from_text(self, text: str):
        reg_exp = r"(\w+)': '([^']+)'"
        try:
            matches = re.findall(reg_exp, text)
            out = dict(matches)
        except:
            out = {}
        return out

    def _compute(self) -> tuple[float, float, float]:
        params_tensor = utils.create_tensor(self.param_preds)
        values_tensor = utils.create_tensor(self.value_preds)
        only_values_tensor = utils.create_tensor(self.value_preds)
        param_acc = torch.mean(params_tensor, dtype=torch.float)
        value_acc = torch.mean(values_tensor, dtype=torch.float)
        only_value_acc = torch.mean(values_tensor, dtype=torch.float)
        return param_acc, value_acc, only_value_acc

    def compute_row(self, pred, ref):
        ref_params = self._get_parameters_from_text(ref)
        pred_params = self._get_parameters_from_text(pred)
        if ref_params == {}:
            if pred_params == {}:
                return 1.0, 1.0, 1.0
            else:
                return 0.0, 0.0, 0.0
        param_accs, value_accs, only_value_accs = [], [], []
        for k, v in ref_params.items():
            only_value_accs.append(self.get_only_value_score(pred_params, v))
            if k in pred_params.keys():
                fuzz_score = utils.fuzzy_string_match(pred_params[k], v)
                param_accs.append(1)
                value_accs.append(fuzz_score)
            else:
                param_accs.append(0)
                value_accs.append(0)

        param_acc = np.mean(param_accs)
        value_acc = np.mean(value_accs)
        ony_value_acc = np.mean(only_value_accs)
        return round(param_acc, 4), round(value_acc, 4), round(ony_value_acc, 4)

    def __str__(self) -> str:
        params, values, only_values = self._compute()
        return f"{self.name} Api Call Parameters Accuracy: {params*100:.2f}, Values Accuracy: {values*100:.2f}, Only Values Accuracy: {only_values*100:.2f}"

    def get_only_value_score(self, pred_params, value):
        score = 0
        for k, v in pred_params.items():
            fuzz_score = utils.fuzzy_string_match(v, value)
            score = max(score, fuzz_score)
        return score
