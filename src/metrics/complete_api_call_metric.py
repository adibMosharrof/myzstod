import torch

import utils
from torchmetrics import Metric
from metrics.tod_metrics_base import TodMetricsBase


class CompleteApiCallMetric(TodMetricsBase):
    def __init__(self):
        super().__init__()
        self.add_state("results", [], dist_reduce_fx="cat")

    def _update(self, method_metrics, params_metrics) -> int:
        res = self.compute_row(method_metrics, params_metrics)
        res = utils.create_tensor(res)
        self.results.append(res)
        return res

    def compute_row(self, method_metrics, params_metrics) -> int:
        method_metric = method_metrics[0]
        params_metric_pair = [params_metrics[0][0], params_metrics[0][1]]

        res = 0
        if method_metric == 1:
            p_res = torch.prod(torch.tensor(params_metric_pair))
            if p_res == 1.0:
                res = 1
        return res

    def _compute(self):
        if not len(self.results):
            return torch.tensor(-1)
            raise ValueError("You must call compute row before calling compute")

        res = torch.mean(utils.create_tensor(self.results), dtype=torch.float)
        return res

    def __str__(self) -> str:
        return f"Complete API Call Accuracy: {self.compute()*100:.2f}"
