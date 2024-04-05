import uuid
from dotmap import DotMap
import evaluate
import pandas as pd

from logger.inference_logger_dataclasses import (
    BertScoreData,
    ApiCallInferenceLogData,
)
from metrics.api_call_invoke_metric import ApiCallInvokeMetric
from metrics.complete_api_call_metric import CompleteApiCallMetric
from metrics.api_call_parameters_metric import ApiCallParametersMetric
from metrics.bert_score_metric import BertScoreMetric
from metrics.nlg_gleu_metric import NlgGleuMetric
from metrics.api_call_method_metric import ApiCallMethodMetric
from torchmetrics import MetricCollection
from accelerate import Accelerator

from my_enums import TurnRowType

accelerator = Accelerator()


class NlgApiCallMetricManager:
    def __init__(self, logger, tokenizer):
        self.tokenizer = tokenizer
        self.google_bleu = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        self.bert_score_metric = evaluate.load(
            "bertscore", experiment_id=str(uuid.uuid4())
        )
        self.logger = logger
        self.data: list[ApiCallInferenceLogData] = []

        self.response_metrics = MetricCollection(
            {
                "response_gleu": NlgGleuMetric(),
                "response_bleu": NlgGleuMetric("bleu"),
                # "response_bertscore": BertScoreMetric(tokenizer),
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

    # @accelerator.on_main_process
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

    def add_batch(
        self,
        input_tokens,
        label_tokens,
        pred_tokens,
        turn_row_types,
        is_retrievals,
        is_slot_fills,
    ):
        input_texts, labels, preds = [
            self.tokenizer.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for tokens in [input_tokens, label_tokens, pred_tokens]
        ]

        response_preds, response_labels, sc_preds, sc_labels = [], [], [], []

        for input_text, pred, label, turn_row_type, is_retrieval, is_slot_fill in zip(
            input_texts, preds, labels, turn_row_types, is_retrievals, is_slot_fills
        ):
            row = ApiCallInferenceLogData(
                input_text=input_text,
                pred=pred,
                label=label,
                turn_row_type=int(turn_row_type),
                is_retrieval=int(is_retrieval),
                is_slot_fill=int(is_slot_fill),
            )
            self.data.append(row)
            if turn_row_type == 0:
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
        for row in self.data:
            row_dict = DotMap(row.__dict__)
            if row.turn_row_type == TurnRowType.RESPONSE.value:
                for k, v in zip(
                    list(self.response_metrics.keys()),
                    list(self.response_metrics.values()),
                ):
                    res = v.compute_row(row.pred, row.label)
                    row_dict[k] = res
            else:
                for k, v in zip(
                    list(self.api_call_metrics.keys()),
                    list(self.api_call_metrics.values()),
                ):
                    res = v.compute_row(row.pred, row.label)
                    if k == "api_call_params":
                        row_dict.api_call_param_names = res[0]
                        row_dict.api_call_param_values = res[1]
                    else:
                        row_dict[k] = res

                row_dict.complete_api_call = self.complete_api_call.compute_row(
                    [row_dict.api_call_method],
                    [(row_dict.api_call_param_names, row_dict.api_call_param_values)],
                )
                self.complete_api_call.update(
                    [row_dict.api_call_method],
                    [(row_dict.api_call_param_names, row_dict.api_call_param_values)],
                )
            row.update(row_dict)

    def compute_is_retrieval_and_slot_fill_metrics(self):

        df = pd.DataFrame(self.data)

        retrievals = df[df.is_retrieval == 1]
        slot_fills = df[df.is_slot_fill == 1]

        r_bleu = retrievals.response_bleu.mean()
        r_gleu = retrievals.response_gleu.mean()
        s_bleu = slot_fills.response_bleu.mean()
        s_gleu = slot_fills.response_gleu.mean()

        self.logger.info(f"Retrieval BLEU: {r_bleu:.4f}")
        self.logger.info(f"Retrieval GLEU: {r_gleu:.4f}")
        self.logger.info(f"Slot Fill BLEU: {s_bleu:.4f}")
        self.logger.info(f"Slot Fill GLEU: {s_gleu:.4f}")
