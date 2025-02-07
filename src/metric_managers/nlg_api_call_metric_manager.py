from dataclasses import asdict
import os
from pathlib import Path
import uuid
from dotmap import DotMap
import evaluate
import pandas as pd

from data_exploration.calc_bert_score import CalcBertScore
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
from utilities.context_manager import ContextManager
import utils

# accelerator = Accelerator()


class NlgApiCallMetricManager:
    def __init__(self, logger, tokenizer=None, cfg=None):
        self.cfg = cfg
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
                "response_bertscore": BertScoreMetric(tokenizer),
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

        self.multi_domain_api_call_metrics = MetricCollection(
            {
                "multi_domain_api_call_method": ApiCallMethodMetric(
                    name="Multi Domain"
                ),
                "multi_domain_api_call_params": ApiCallParametersMetric(
                    name="Multi Domain"
                ),
                "multi_domain_api_call_invoke": ApiCallInvokeMetric(
                    invoke_text="ApiCall", name="Multi Domain "
                ),
            }
        )
        self.multi_domain_complete_api_call = CompleteApiCallMetric(name="Multi Domain")

    # @accelerator.on_main_process
    def compute_metrics(self, domain_names: str):
        all_metrics = (
            list(self.response_metrics.values())
            + list(self.api_call_metrics.values())
            + [self.complete_api_call]
            + list(self.multi_domain_api_call_metrics.values())
            + [self.multi_domain_complete_api_call]
        )
        for v in all_metrics:
            res = str(v)
            utils.log(self.logger, res)
            # self.logger.info(res)
            # print(res)

    def add_batch(
        self,
        input_tokens,
        label_tokens,
        pred_tokens,
        turn_row_types,
        is_retrievals,
        is_slot_fills,
        dialog_ids,
        turn_ids,
        is_multi_domain_api_calls,
        domains,
        is_single_domains,
        current_user_utterances,
        search_results,
    ):
        if self.tokenizer:
            input_texts, labels, preds = [
                self.tokenizer.batch_decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for tokens in [input_tokens, label_tokens, pred_tokens]
            ]
        else:
            input_texts, labels, preds = input_tokens, label_tokens, pred_tokens

        (
            response_preds,
            response_labels,
            api_preds,
            api_labels,
            multi_api_preds,
            multi_api_labels,
        ) = ([], [], [], [], [], [])

        for (
            input_text,
            pred,
            label,
            turn_row_type,
            is_retrieval,
            is_slot_fill,
            dialog_id,
            turn_id,
            is_multi_domain_api_call,
            domain,
            is_single_domain,
            current_user_utterance,
            search_result,
        ) in zip(
            input_texts,
            preds,
            labels,
            turn_row_types,
            is_retrievals,
            is_slot_fills,
            dialog_ids,
            turn_ids,
            is_multi_domain_api_calls,
            domains,
            is_single_domains,
            current_user_utterances,
            search_results,
        ):
            row = ApiCallInferenceLogData(
                input_text=input_text,
                pred=pred,
                label=label,
                turn_row_type=int(turn_row_type),
                is_retrieval=int(is_retrieval),
                is_slot_fill=int(is_slot_fill),
                dialog_id=dialog_id.item(),
                turn_id=turn_id.item(),
                domains=domain,
                is_multi_domain_api_call=int(is_multi_domain_api_call),
                is_single_domain=int(is_single_domain),
                current_user_utterance=current_user_utterance,
                search_results=search_result,
            )
            self.data.append(row)
            if turn_row_type == 0:
                response_preds.append(row.pred)
                response_labels.append(row.label)
            else:
                api_preds.append(row.pred)
                api_labels.append(row.label)
                if is_multi_domain_api_call:
                    multi_api_preds.append(row.pred)
                    multi_api_labels.append(row.label)
        self.response_metrics.update(
            references=response_labels, predictions=response_preds
        )
        self.api_call_metrics.update(references=api_labels, predictions=api_preds)
        self.multi_domain_api_call_metrics.update(
            references=multi_api_labels, predictions=multi_api_preds
        )

    def write_csv(self, csv_path):
        if not len(self.data):
            raise ValueError("Must call compute row wise metrics first")
        df = pd.DataFrame([asdict(d) for d in self.data])
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
                        # row_dict.api_call_only_param_values = res[2]
                        if len(res) == 3:
                            row_dict.api_call_param_relation = res[2]
                    else:
                        row_dict[k] = res

                row_dict.complete_api_call = self.complete_api_call.compute_row(
                    [row_dict.api_call_method],
                    [
                        (
                            row_dict.api_call_param_names,
                            row_dict.api_call_param_values,
                        )
                    ],
                )
                self.complete_api_call.update(
                    [row_dict.api_call_method],
                    [
                        (
                            row_dict.api_call_param_names,
                            row_dict.api_call_param_values,
                        )
                    ],
                )
                if row.is_multi_domain_api_call:
                    for k, v in zip(
                        list(self.multi_domain_api_call_metrics.keys()),
                        list(self.multi_domain_api_call_metrics.values()),
                    ):
                        res = v.compute_row(row.pred, row.label)
                        if "api_call_params" in k:
                            row_dict.multi_domain_api_call_param_names = res[0]
                            row_dict.multi_domain_api_call_param_values = res[1]
                            row_dict.multi_domain_api_call_only_param_values = res[2]
                            if len(res) == 4:
                                row_dict.multi_domain_api_call_param_relation = res[3]
                        else:
                            row_dict[k] = res

                    row_dict.multi_domain_complete_api_call = (
                        self.multi_domain_complete_api_call.compute_row(
                            [row_dict.multi_domain_api_call_method],
                            [
                                (
                                    row_dict.multi_domain_api_call_param_names,
                                    row_dict.multi_domain_api_call_param_values,
                                )
                            ],
                        )
                    )
                    self.multi_domain_complete_api_call.update(
                        [row_dict.multi_domain_api_call_method],
                        [
                            (
                                row_dict.multi_domain_api_call_param_names,
                                row_dict.multi_domain_api_call_param_values,
                            )
                        ],
                    )
            self.hook_for_additional_metrics(row, row_dict)
            row.update(row_dict)

    def hook_for_additional_metrics(self, row, row_dict):
        pass

    def compute_is_retrieval_and_slot_fill_metrics(self):

        df = pd.DataFrame(self.data)

        retrievals = df[df.is_retrieval == 1]
        slot_fills = df[df.is_slot_fill == 1]

        r_bleu = retrievals.response_bleu.mean()
        r_gleu = retrievals.response_gleu.mean()
        s_bleu = slot_fills.response_bleu.mean()
        s_gleu = slot_fills.response_gleu.mean()

        utils.log(self.logger, f"Retrieval BLEU: {r_bleu:.4f}")
        utils.log(self.logger, f"Retrieval GLEU: {r_gleu:.4f}")
        utils.log(self.logger, f"Slot Fill BLEU: {s_bleu:.4f}")
        utils.log(self.logger, f"Slot Fill GLEU: {s_gleu:.4f}")

    def get_dataset_from_cfg(self):
        for key in ["ketod", "sgd", "bitod"]:
            if key in self.cfg.dataset:
                return self.cfg.dataset[key]
        return None

    def compute_bert_scores(self):
        dataset = self.get_dataset_from_cfg()
        out_path = Path(os.getcwd()) / "results" / "bertscores.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cbs = CalcBertScore(
            DotMap(
                data=self.data,
                out_path=out_path,
                project_root=self.cfg.project_root,
                raw_data_root=dataset["raw_data_root"],
            )
        )
        cbs.run()
