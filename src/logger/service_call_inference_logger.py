from dataclasses import dataclass
from logger.inference_logger_dataclasses import (
    BertScoreData,
    ServiceCallInferenceLogData,
)
import utils
import numpy as np
import pandas as pd


class ServiceCallInferenceLogger:
    def __init__(self, tokenizer, metric_manager):
        self.concat_labels = None
        self.concat_preds = None
        self.tokenizer = tokenizer
        self.metric_manager = metric_manager
        self.data: list[ServiceCallInferenceLogData] = []

    def add_batch(self, input_tokens, label_tokens, pred_tokens, service_calls):
        input_texts, preds, labels = [
            self.tokenizer.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for tokens in [input_tokens, label_tokens, pred_tokens]
        ]

        self.data += [
            ServiceCallInferenceLogData(input_text=i, label=l, pred=p, is_api_call=s)
            for i, p, l, s in zip(input_texts, preds, labels, service_calls)
        ]

        for d in self.data:
            gleu_score, bert_score_data = self.metric_manager.compute_single_row(
                d.pred, d.label
            )
            if d.is_api_call:
                d.service_call_gleu_score = gleu_score
                d.service_call_bert_score = bert_score_data
            else:
                d.response_gleu_score = gleu_score
                d.response_bert_score = bert_score_data

    def write_csv(self, csv_path):
        return
        self.concat_labels = np.concatenate(self.all_labels, axis=0)
        self.concat_preds = np.concatenate(self.all_preds, axis=0)
        concat_input_texts = np.concatenate(self.all_input_texts, axis=0)
        return
        df = pd.DataFrame(
            {
                "input_texts": concat_input_texts,
                "target_text": self.concat_labels,
                "pred_text": self.concat_preds,
                "gleu_score": self.all_gleu_scores,
                "response_gleu_score": self.all_response_gleu_scores,
                "service_call_gleu_score": self.all_service_call_gleu_scores,
                "bs_precision": self.all_bert_score_precision,
                "bs_recall": self.all_bert_score_recall,
                "bs_f1": self.all_bert_score_f1,
            }
        )

        df.to_csv(csv_path, index=False, encoding="utf-8")
