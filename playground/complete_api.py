from dotmap import DotMap
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))
import utils
from metrics.complete_api_call_metric import CompleteApiCallMetric
from metrics.api_call_parameters_metric import ApiCallParametersMetric
from logger.inference_logger_dataclasses import ApiCallInferenceLogData
from metrics.api_call_method_metric import ApiCallMethodMetric


class CompleteApi:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.api_call_metrics ={
        #         "api_call_method": ApiCallMethodMetric(),
        #         "api_call_params": ApiCallParametersMetric(),
        #     }

        # self.complete_api_call = CompleteApiCallMetric()

    def run(self):
        for path in self.cfg.out_files:
            self.api_call_metrics = {
                "api_call_method": ApiCallMethodMetric(),
                "api_call_params": ApiCallParametersMetric(),
            }

            self.complete_api_call = CompleteApiCallMetric()
            all_data: list[ApiCallInferenceLogData] = utils.read_csv_dataclass(
                path, ApiCallInferenceLogData
            )
            api_call_data = [d for d in all_data if d.is_api_call]

            metric_objects = list(self.api_call_metrics.values())
            metric_names = list(self.api_call_metrics.keys())
            for row in api_call_data:
                row_dict = DotMap(row.__dict__)
                for k, v in zip(metric_names, metric_objects):
                    res = v.compute_row(row_dict.pred, row_dict.label)
                    row_dict[k] = res
                row_dict.complete_api_call = self.complete_api_call.update(
                    [row_dict.api_call_method], [row_dict.api_call_params]
                )
                row = ApiCallInferenceLogData(**row_dict)
            print(str(self.complete_api_call))
        a = 1


if __name__ == "__main__":
    ca = CompleteApi(
        DotMap(
            out_files=[
                "playground/data/all.csv",
                "playground/data/seen.csv",
                "playground/data/unseen.csv",
            ]
        )
    )
    ca.run()
