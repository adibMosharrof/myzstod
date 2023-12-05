import numpy as np


class CompleteApiCallMetric:
    def __init__(self):
        self.results = []

    def compute_row(self, method_metric, params_metric) -> int:
        res = 1
        if not method_metric or not params_metric:
            res = 0
        self.results.append(res)
        return res

    def compute(self):
        if not (self.results):
            raise ValueError("You must call compute row before calling compute")
        res = np.mean(self.results)
        return res

    def __str__(self) -> str:
        return f"Complete API Call Accuracy: {self.compute()*100:.2f}"
