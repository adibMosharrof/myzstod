import re
from typing import Optional
import uuid
import evaluate
import numpy as np
import pandas as pd
from prettytable import PrettyTable

from logger.inference_logger import InferenceLogger
from logger.inference_logger_dataclasses import (
    BertScoreData,
    ServiceCallInferenceLogData,
)
from logger.service_call_inference_logger import ServiceCallInferenceLogger
import utils


class NlgServiceCallMetricManager:
    def __init__(self, logger, tokenizer):
        self.tokenizer = tokenizer
        self.google_bleu = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        self.bert_score_metric = evaluate.load(
            "bertscore", experiment_id=str(uuid.uuid4())
        )
        self.bert_score_model = "distilbert-base-uncased"
        self.logger = logger
        self.data: list[ServiceCallInferenceLogData] = []

    def compute_bert_score(self, predictions, references):
        bert_score_result = self.bert_score_metric.compute(
            predictions=predictions,
            references=references,
            model_type=self.bert_score_model,
        )
        avg_precision = np.mean(bert_score_result["precision"])
        avg_recall = np.mean(bert_score_result["recall"])
        avg_f1 = np.mean(bert_score_result["f1"])
        bert_score_data = BertScoreData(
            precision=avg_precision, recall=avg_recall, f1=avg_f1
        )
        return bert_score_data

    def compute_gleu_score(self, predictions, references):
        refs = np.expand_dims(references, axis=1)
        gleu_result = self.google_bleu.compute(predictions=predictions, references=refs)
        gleu_score = gleu_result["google_bleu"]
        return gleu_score

    def compute_metrics(self):
        pt = PrettyTable()
        pt.field_names = [
            "Setting",
            "GLEU",
            "BERT Precision",
            "BERT Recall",
            "BERT F1",
            "Api Call Score",
        ]
        df = pd.DataFrame(self.data)

        response_rows = df[df.is_service_call == 0]

        response_gleu_result = self.compute_gleu_score(
            response_rows.pred.values, response_rows.label.values
        )
        response_bert_score_data = self.compute_bert_score(
            response_rows.pred.values, response_rows.label.values
        )
        # empty_api_call_rows = [0] * len(response_rows)
        pt.add_row(
            [
                "Response",
                response_gleu_result,
                response_bert_score_data.precision,
                response_bert_score_data.recall,
                response_bert_score_data.f1,
                -1,
            ]
        )

        service_call_rows = df[df.is_service_call == 1]
        service_call_preds = service_call_rows.pred.values

        service_call_gleu_result = self.compute_gleu_score(
            predictions=service_call_preds, references=service_call_rows.label.values
        )
        service_call_bert_score_data = self.compute_bert_score(
            service_call_preds, service_call_rows.label.values
        )
        scores = [
            self.compute_api_call_score(p, l)
            for p, l in zip(service_call_preds, service_call_rows.label.values)
        ]
        api_call_score = np.mean(scores)
        pt.add_row(
            [
                "Service Call",
                service_call_gleu_result,
                service_call_bert_score_data.precision,
                service_call_bert_score_data.recall,
                service_call_bert_score_data.f1,
                api_call_score,
            ]
        )

        all_preds = df.pred.values
        all_gleu_result = self.compute_gleu_score(
            predictions=all_preds, references=df.label.values
        )
        all_bert_score_data = self.compute_bert_score(
            service_call_preds, service_call_rows.label.values
        )
        pt.add_row(
            [
                "Overall",
                all_gleu_result,
                all_bert_score_data.precision,
                all_bert_score_data.recall,
                all_bert_score_data.f1,
                -1,
            ]
        )
        self.logger.info(pt)
        print(pt)
        score_str = f"Api call score {api_call_score}"
        self.logger.info(score_str)
        print(score_str)

    def compute_single_row(self, preds, labels):
        bleu_result = self.google_bleu.compute(
            predictions=[preds], references=[[labels]]
        )
        bert_score_data = self.compute_bert_score([preds], [labels])
        return (round(bleu_result["google_bleu"], 4), bert_score_data)

    def compute_api_call_score(self, preds, labels) -> Optional[float]:
        regexp = r"ServiceCall\((.*?)\)"
        label_api_calls = re.findall(regexp, labels)
        if not len(label_api_calls):
            raise ValueError("No api calls in label")

        pred_api_calls = re.findall(regexp, preds)
        if not len(pred_api_calls):
            return 0
        score = utils.fuzzy_string_match(pred_api_calls, label_api_calls) / 100.0
        return score

    def add_batch(self, input_tokens, label_tokens, pred_tokens, service_calls):
        input_texts, labels, preds = [
            self.tokenizer.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for tokens in [input_tokens, label_tokens, pred_tokens]
        ]

        self.data += [
            ServiceCallInferenceLogData(
                input_text=i, label=l, pred=p, is_service_call=int(s)
            )
            for i, p, l, s in zip(input_texts, preds, labels, service_calls)
        ]

        for d in self.data:
            gleu_score, bert_score_data = self.compute_single_row(d.pred, d.label)
            d.gleu_score = gleu_score
            d.bert_score_data = str(bert_score_data)

    def write_csv(self, csv_path):
        df = pd.DataFrame(self.data)
        df.to_csv(csv_path, index=False, encoding="utf-8")
