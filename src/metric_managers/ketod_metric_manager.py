import uuid
from dotmap import DotMap
import evaluate
import pandas as pd

from logger.inference_logger_dataclasses import (
    BertScoreData,
    ApiCallInferenceLogData,
    KetodInferenceLogData,
)
from metrics.api_call_invoke_metric import ApiCallInvokeMetric
from metrics.complete_api_call_metric import CompleteApiCallMetric
from metrics.api_call_parameters_metric import ApiCallParametersMetric
from metrics.bert_score_metric import BertScoreMetric
from metrics.nlg_gleu_metric import NlgGleuMetric
from metrics.api_call_method_metric import ApiCallMethodMetric
from torchmetrics import MetricCollection
from accelerate import Accelerator
import utils
from my_enums import TurnRowType

accelerator = Accelerator()


class KeTodMetricManager:
    def __init__(self, logger, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logger
        self.data: list[ApiCallInferenceLogData] = []

        self.response_metrics = MetricCollection(
            {
                "response_gleu": NlgGleuMetric(),
                "response_bleu": NlgGleuMetric("bleu"),
            }
        )
        self.api_call_metrics = MetricCollection(
            {
                "api_call_method": ApiCallMethodMetric(),
                "api_call_params": ApiCallParametersMetric(),
                "api_call_invoke": ApiCallInvokeMetric(invoke_text="ApiCall"),
            }
        )
        self.complete_api_call = CompleteApiCallMetric()

        self.ke_metrics = MetricCollection(
            {
                "ke_method": ApiCallMethodMetric(name="ke"),
                "ke_params": ApiCallParametersMetric(name="ke"),
                "ke_api_call_invoke": ApiCallInvokeMetric(invoke_text="EntityQuery"),
            }
        )
        self.complete_kb_call = CompleteApiCallMetric()

    def compute_metrics(self, domain_names: str):
        all_metrics = (
            list(self.response_metrics.values())
            + list(self.ke_metrics.values())
            + list(self.api_call_metrics.values())
            + [self.complete_api_call, self.complete_kb_call]
        )
        for v in all_metrics:
            res = str(v)
            utils.log(self.logger,res)
            # print(res)

    def add_batch(self, input_tokens, label_tokens, pred_tokens, turn_row_type):
        input_texts, labels, preds = [
            self.tokenizer.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for tokens in [input_tokens, label_tokens, pred_tokens]
        ]

        response_preds, response_labels = [], []
        sc_preds, sc_labels = [], []
        ke_preds, ke_labels = [], []

        for i, p, l, s in zip(input_texts, preds, labels, turn_row_type):
            row = KetodInferenceLogData(i, l, p, int(s))
            self.data.append(row)
            if s == TurnRowType.RESPONSE.value:
                response_preds.append(row.pred)
                response_labels.append(row.label)
            elif s == TurnRowType.API_CALL.value:
                sc_preds.append(row.pred)
                sc_labels.append(row.label)
            elif s == TurnRowType.KE_QUERY.value:
                ke_preds.append(row.pred)
                ke_labels.append(row.label)
            else:
                raise ValueError(f"Unknown turn row type {s}")
        self.response_metrics.update(
            references=response_labels, predictions=response_preds
        )
        self.api_call_metrics.update(references=sc_labels, predictions=sc_preds)
        self.ke_metrics.update(references=ke_labels, predictions=ke_preds)

    def write_csv(self, csv_path):
        if not len(self.data):
            raise ValueError("Must call compute row wise metrics first")
        df = pd.DataFrame(self.data)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    def compute_row_wise_metrics(self):
        for row in self.data:
            row_dict = DotMap(row.__dict__)
            if row.turn_row_type == TurnRowType.RESPONSE.value:
                for k, v in zip(
                    list(self.response_metrics.keys()),
                    list(self.response_metrics.values()),
                ):
                    res = v.compute_row(row.pred, row.label)
                    row_dict[k] = res
            elif row.turn_row_type == TurnRowType.API_CALL.value:
                for k, v in zip(
                    list(self.api_call_metrics.keys()),
                    list(self.api_call_metrics.values()),
                ):
                    res = v.compute_row(row.pred, row.label)
                    row_dict[k] = res
                row_dict.complete_api_call = self.complete_api_call.update(
                    [row_dict.api_call_method], [row_dict.api_call_params]
                )
            elif row.turn_row_type == TurnRowType.KE_QUERY.value:
                for k, v in zip(
                    list(self.ke_metrics.keys()),
                    list(self.ke_metrics.values()),
                ):
                    res = v.compute_row(row.pred, row.label)
                    row_dict[k] = res
                row_dict.complete_kb_call = self.complete_kb_call.update(
                    [row_dict.ke_method], [row_dict.ke_params]
                )

            row = KetodInferenceLogData(**row_dict)
        # metric_objects = list(self.response_metrics.values()) + list(
        #     self.api_call_metrics.values()
        # )
        # metric_names = list(self.response_metrics.keys()) + list(
        #     self.api_call_metrics.keys()
        # )

        # for row in self.data:
        #     row_dict = DotMap(row.__dict__)
        #     for k, v in zip(metric_names, metric_objects):
        #         res = v.compute_row(row_dict.pred, row_dict.label)
        #         row_dict[k] = res
        #     row_dict.complete_api_call = self.complete_api_call.compute_row(
        #         row_dict.api_call_method, row_dict.api_call_params
        #     )
        #     row = ServiceCallInferenceLogData(**row_dict)
