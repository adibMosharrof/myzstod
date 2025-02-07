import uuid
from dotmap import DotMap
import evaluate
import pandas as pd

from logger.inference_logger_dataclasses import (
    BertScoreData,
    ApiCallInferenceLogData,
    KetodInferenceLogData,
)
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
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

# accelerator = Accelerator()


class KeTodMetricManager(NlgApiCallMetricManager):
    def __init__(self, logger, tokenizer, cfg):
        super().__init__(logger, tokenizer, cfg)

        self.ke_metrics = MetricCollection(
            {
                "ke_method": ApiCallMethodMetric(name="ke"),
                "ke_params": ApiCallParametersMetric(name="ke"),
                "ke_api_call_invoke": ApiCallInvokeMetric(invoke_text="EntityQuery"),
            }
        )
        self.complete_kb_call = CompleteApiCallMetric(name="KE")

    def compute_metrics(self, domain_names: str):
        all_metrics = (
            list(self.response_metrics.values())
            + list(self.ke_metrics.values())
            + list(self.api_call_metrics.values())
            + [self.complete_api_call, self.complete_kb_call]
            + list(self.multi_domain_api_call_metrics.values())
            + [self.multi_domain_complete_api_call]
        )
        for v in all_metrics:
            res = str(v)
            utils.log(self.logger, res)
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
        input_texts, labels, preds = [
            self.tokenizer.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for tokens in [input_tokens, label_tokens, pred_tokens]
        ]

        # response_preds, response_labels = [], []
        # sc_preds, sc_labels = [], []
        # ke_preds, ke_labels = [], []
        (
            response_preds,
            response_labels,
            api_preds,
            api_labels,
            multi_api_preds,
            multi_api_labels,
            ke_preds,
            ke_labels,
        ) = ([], [], [], [], [], [], [], [])
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
            row = KetodInferenceLogData(
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
            if turn_row_type == TurnRowType.RESPONSE.value:
                response_preds.append(row.pred)
                response_labels.append(row.label)
            elif turn_row_type == TurnRowType.API_CALL.value:
                api_preds.append(row.pred)
                api_labels.append(row.label)
                if is_multi_domain_api_call:
                    multi_api_preds.append(row.pred)
                    multi_api_labels.append(row.label)
            elif turn_row_type == TurnRowType.KE_QUERY.value:
                ke_preds.append(row.pred)
                ke_labels.append(row.label)
        self.response_metrics.update(
            references=response_labels, predictions=response_preds
        )
        self.api_call_metrics.update(references=api_labels, predictions=api_preds)
        self.multi_domain_api_call_metrics.update(
            references=multi_api_labels, predictions=multi_api_preds
        )
        self.ke_metrics.update(references=ke_labels, predictions=ke_preds)

    def hook_for_additional_metrics(self, row, row_dict):
        if row.turn_row_type == TurnRowType.KE_QUERY.value:
            for k, v in zip(
                list(self.ke_metrics.keys()),
                list(self.ke_metrics.values()),
            ):
                res = v.compute_row(row.pred, row.label)
                if k == "ke_params":
                    row_dict.ke_params = res[0]
                    row_dict.ke_param_values = res[1]
                else:
                    row_dict[k] = res
            row_dict.complete_kb_call = self.complete_kb_call.compute_row(
                [row_dict.ke_method],
                [
                    (
                        row_dict.ke_params,
                        row_dict.ke_param_values,
                    )
                ],
            )
            self.complete_kb_call.update(
                [row_dict.ke_method],
                [
                    (
                        row_dict.ke_params,
                        row_dict.ke_param_values,
                    )
                ],
            )
