import evaluate
from torchmetrics import MetricCollection
import uuid
from logger.inference_logger_dataclasses import ApiCallInferenceLogData
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from metrics.api_call_invoke_metric import ApiCallInvokeMetric
from metrics.api_call_method_metric import ApiCallMethodMetric
from metrics.bitod_api_call_parameters_metric import BitodApiCallParametersMetric
from metrics.bitod_complete_api_call_metric import BitodCompleteApiCallMetric
from metrics.complete_api_call_metric import CompleteApiCallMetric
from metrics.nlg_gleu_metric import NlgGleuMetric


class BitodMetricManager(NlgApiCallMetricManager):
    def __init__(self, logger, tokenizer):
        super().__init__(logger, tokenizer)

        self.response_metrics = MetricCollection(
            {
                "response_gleu": NlgGleuMetric(),
                "response_bleu": NlgGleuMetric("bleu"),
            }
        )
        self.api_call_metrics = MetricCollection(
            {
                "api_call_params": BitodApiCallParametersMetric(),
                "api_call_method": ApiCallMethodMetric(reg_exp=r"method=([^\s,]+)"),
                "api_call_invoke": ApiCallInvokeMetric(invoke_text="ApiCall"),
            }
        )
        self.complete_api_call = BitodCompleteApiCallMetric()
