import uuid
from dotmap import DotMap
import evaluate
import pandas as pd

from logger.inference_logger_dataclasses import (
    BertScoreData,
    ServiceCallInferenceLogData,
)
from metrics.complete_api_call_metric import CompleteApiCallMetric
from metrics.api_call_parameters_metric import ApiCallParametersMetric
from metrics.bert_score_metric import BertScoreMetric
from metrics.nlg_gleu_metric import NlgGleuMetric
from metrics.api_call_method_metric import ApiCallMethodMetric
from torchmetrics import MetricCollection
from accelerate import Accelerator

accelerator = Accelerator()


class NlgApiCallMetricManager:
    def __init__(self, logger, tokenizer):
        self.tokenizer = tokenizer
        self.google_bleu = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        self.bert_score_metric = evaluate.load(
            "bertscore", experiment_id=str(uuid.uuid4())
        )
        self.logger = logger
        self.data: list[ServiceCallInferenceLogData] = []

        self.response_metrics = MetricCollection(
            {
                "response_gleu": NlgGleuMetric(tokenizer),
                # "response_bertscore": BertScoreMetric(tokenizer),
            }
        )
        self.api_call_metrics = MetricCollection(
            {
                "api_call_method": ApiCallMethodMetric(),
                "api_call_params": ApiCallParametersMetric(),
            }
        )
        self.complete_api_call = CompleteApiCallMetric()

    @accelerator.on_main_process
    def compute_metrics(self, domain_names: str):
        all_metrics = (
            list(self.response_metrics.values())
            + list(self.api_call_metrics.values())
            + [self.complete_api_call]
        )
        for v in all_metrics:
            res = str(v)
            self.logger.info(res)
            print(res)

    def add_batch(self, input_tokens, label_tokens, pred_tokens, api_calls):
        input_texts, labels, preds = [
            self.tokenizer.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for tokens in [input_tokens, label_tokens, pred_tokens]
        ]

        response_preds, response_labels, sc_preds, sc_labels = [], [], [], []

        for i, p, l, s in zip(input_texts, preds, labels, api_calls):
            row = ServiceCallInferenceLogData(i, l, p, int(s))
            self.data.append(row)
            if s == 0:
                response_preds.append(row.pred)
                response_labels.append(row.label)
            else:
                sc_preds.append(row.pred)
                sc_labels.append(row.label)
        self.response_metrics.update(
            references=response_labels, predictions=response_preds
        )
        self.api_call_metrics.update(references=sc_labels, predictions=sc_preds)

    def write_csv(self, csv_path):
        if not len(self.data):
            raise ValueError("Must call compute row wise metrics first")
        df = pd.DataFrame(self.data)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    def compute_row_wise_metrics(self):
        metric_objects = list(self.response_metrics.values()) + list(
            self.api_call_metrics.values()
        )
        metric_names = list(self.response_metrics.keys()) + list(
            self.api_call_metrics.keys()
        )
        for row in self.data:
            row_dict = DotMap(row.__dict__)
            for k, v in zip(metric_names, metric_objects):
                res = v.compute_row(row_dict.pred, row_dict.label)
                row_dict[k] = res
            row_dict.complete_api_call = self.complete_api_call.compute_row(
                row_dict.api_call_method, row_dict.api_call_params
            )
            row = ServiceCallInferenceLogData(**row_dict)
