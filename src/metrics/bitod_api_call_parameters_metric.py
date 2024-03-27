import torch
from metrics.tod_metrics_base import TodMetricsBase
import re
from tod.nlg.bitod_api_call import BitodApiCallParams
import utils
import numpy as np


class BitodApiCallParametersMetric(TodMetricsBase):
    def __init__(self, name: str = ""):
        super().__init__()
        self.add_state("param_accs", [], dist_reduce_fx="cat")
        self.add_state("relation_accs", [], dist_reduce_fx="cat")
        self.add_state("value_accs", [], dist_reduce_fx="cat")
        self.name = name

    def _update(self, predictions: list[str], references: list[str]) -> None:
        for pred, ref in zip(predictions, references):
            param_acc, relation_acc, value_acc = self.compute_row(pred, ref)
            self.param_accs.append(param_acc)
            self.relation_accs.append(relation_acc)
            self.value_accs.append(value_acc)

    def _get_parameters_from_text(self, text: str) -> list[BitodApiCallParams]:
        # reg_exp = r"\{([^}]*)\}"
        # try:
        #     matches = re.findall(reg_exp, text)
        #     reg_out = matches[0].split("|")
        # except:
        #     reg_out = {}

        parameters_index = text.find("parameters=")
        if parameters_index == -1:
            return []

        parameters_content = text[parameters_index + len("parameters=") :].strip(")")
        if not parameters_content:
            return []
        parameters = parameters_content.split("|")
        out = []
        items_num = range(3)
        for item in parameters:
            item = item.split(" ")
            for i in items_num:
                if i >= 0 and i < len(item):
                    item.append("")
            out.append(
                BitodApiCallParams(slot_name=item[0], relation=item[1], value=item[2:])
            )
        return out

    def _compute(self) -> tuple[float, float]:
        param_acc = torch.mean(
            utils.create_tensor(self.param_accs, torch.float), dtype=torch.float
        )
        relation_acc = torch.mean(
            utils.create_tensor(self.relation_accs, torch.float), dtype=torch.float
        )
        value_acc = torch.mean(
            utils.create_tensor(self.value_accs, torch.float), dtype=torch.float
        )
        return (
            torch.round(param_acc, decimals=4),
            torch.round(relation_acc, decimals=4),
            torch.round(value_acc, decimals=4),
        )

    def compute_row(self, pred, ref):
        ref_params = self._get_parameters_from_text(ref)
        pred_params = self._get_parameters_from_text(pred)
        if not ref_params:
            if not pred_params:
                t_1 = utils.create_tensor(1.0)
                return t_1, t_1, t_1
            else:
                t_0 = utils.create_tensor(0.0)
                return t_0, t_0, t_0
        param_accs, relation_accs, value_accs = [], [], []
        for ref in ref_params:
            pred = ref.get_by_slot_name(pred_params)
            if pred:
                param_accs.append(utils.create_tensor(1))
                relation_accs.append(utils.create_tensor(ref.relation == pred.relation))
                fuzz_score = utils.fuzzy_string_match(pred.value, ref.value)
                value_accs.append(utils.create_tensor(fuzz_score))
            else:
                param_accs.append(utils.create_tensor(0))
                relation_accs.append(utils.create_tensor(0))
                value_accs.append(utils.create_tensor(0))
        param_acc = torch.mean(utils.create_tensor(param_accs), dtype=torch.float)
        relation_acc = torch.mean(utils.create_tensor(relation_accs), dtype=torch.float)
        value_acc = torch.mean(utils.create_tensor(value_accs), dtype=torch.float)
        return (
            torch.round(param_acc, decimals=4),
            torch.round(relation_acc, decimals=4),
            torch.round(value_acc, decimals=4),
        )

    def __str__(self) -> str:
        params, relations, values = self._compute()
        return f"{self.name} Api Call Parameters Accuracy: {params*100:.2f}, Relations Accuracy: {relations*100:.2f} Values Accuracy: {values*100:.2f}"
