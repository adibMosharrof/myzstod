import torch

import utils
from torchmetrics import Metric
from metrics.tod_metrics_base import TodMetricsBase


class CompleteApiCallMetric(TodMetricsBase):
    def __init__(self):
        super().__init__()
        # self.results = []
        self.add_state("results", [], dist_reduce_fx="cat")

    def _update(self, method_metrics, params_metrics) -> int:
        method_metric = method_metrics[0]
        params_metric = params_metrics[0]
        res = utils.create_tensor(0)
        if method_metric == 1:
            if params_metric[0] * params_metric[1] == 1.0:
                self.results.append(utils.create_tensor(1))
                return res
        self.results.append(res)
        return res

    def _compute(self):
        if not len(self.results):
            raise ValueError("You must call compute row before calling compute")

        res = torch.mean(utils.create_tensor(self.results), dtype=torch.float)
        return res

    def __str__(self) -> str:
        return f"Complete API Call Accuracy: {self.compute()*100:.2f}"
