import os
from pathlib import Path
import pandas as pd
from torchmetrics import MetricCollection
from interpret.interpret_features import InterpretFeatureGroup
from logger.inference_logger_dataclasses import ApiCallInferenceLogData
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from transformers import AutoTokenizer

from metrics.api_call_method_metric import ApiCallMethodMetric
from metrics.api_call_parameters_metric import ApiCallParametersMetric
from metrics.interpret_params_metric import InterpretParamsMetric
from my_enums import InterpretFeatureTypes
import utils


class InterpretMetricManager(NlgApiCallMetricManager):
    def __init__(self, logger, tokenizer=None, cfg=None):
        self.cfg = cfg
        self.tokenizer: AutoTokenizer = tokenizer

        self.logger = logger
        self.data: list[ApiCallInferenceLogData] = []
        self.api_call_metrics = None
        self.feature_name = None
        self.group_name = None
        self.intervention_name = None
        self.interpret_layer = None

    def compute_metrics(self, pred_tokens, label_tokens):
        preds = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
        metric_results = []
        for group, group_metrics in self.api_call_metrics.items():
            for metric in list(group_metrics.values()):
                metric._update(preds, labels)
                # res = f"{self.feature_name} {str(metric)}"
                res = metric.interpret_repr()
                # utils.log(self.logger, res)
                metric_results.extend(res)
        out_dir = "results" / Path(self.group_name) / f"layer_{self.interpret_layer}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{self.intervention_name}.csv"
        # out_file = out_dir / "interventions.csv"
        # with open(out_file, "a") as f:
        #     f.write("\n".join(metric_results))
        out = []
        for result in metric_results:
            out.append(
                {
                    "Layer": self.interpret_layer,
                    "Intervention": self.intervention_name,
                    "Group": self.group_name,
                    "Feature": self.feature_name,
                    "Metric": result["name"],
                    "Value": result["value"],
                }
            )
        df = pd.DataFrame(out)
        df.to_csv(out_file, mode="a", index=False, header=not os.path.exists(out_file))

    def get_metric(self, feature_type, feature_name):
        if feature_type == InterpretFeatureTypes.PARAM.value:
            return MetricCollection(
                {
                    "params": InterpretParamsMetric(target_param=feature_name),
                }
            )
        if feature_type == InterpretFeatureTypes.METHOD.value:
            return MetricCollection(
                {
                    "method": ApiCallMethodMetric(),
                }
            )

    def initialize(
        self,
        feature_types,
        feature_name,
        group_name,
        intervention_name,
        interpret_layer,
    ):
        # for feature_name in interpret_features.features:
        self.api_call_metrics = {
            feature_type: self.get_metric(feature_type, feature_name)
            for feature_type in feature_types
        }
        self.feature_name = feature_name
        self.group_name = group_name
        self.intervention_name = intervention_name
        self.interpret_layer = interpret_layer
