from dotmap import DotMap
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))
from torchmetrics import MetricCollection

from metrics.api_call_invoke_metric import ApiCallInvokeMetric
from metrics.api_call_method_metric import ApiCallMethodMetric
from metrics.api_call_parameters_metric import ApiCallParametersMetric
from metrics.complete_api_call_metric import CompleteApiCallMetric
from metrics.nlg_gleu_metric import NlgGleuMetric


from logger.inference_logger_dataclasses import ApiCallInferenceLogData


class ChatGptInference:
    def __init__(self, cfg):
        self.cfg = cfg
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

    def get_api_call_results(self, row):
        pass

    def get_bleu_results(self, row):
        pass

    def run(self):
        csv_file = pd.read_csv(self.cfg.path)
        output = []
        for i, item in csv_file.iterrows():
            # dict below follows ApiCallInferenceLogData
            out = DotMap(
                input_text=item.context,
                pred=item.response,
                label=item.target,
                turn_row_type=item.api_call,
                is_retrieval=item.retrieval,
                is_slot_fill=item.slot_fill,
                dialog_id=item.dialog_id,
                turn_id=item.turn_id,
                domains=item.domains_original,
            )
            if item.api_call:
                # metrics = self.get_api_call_results(item)
                # out.api_call_method = metrics.api_call_method
                # out.api_call_invoke = metrics.api_call_invoke
                # out.api_call_param_names = metrics.api_call_param_names
                # out.complete_api_call = metrics.complete_api_call
                for k, v in zip(
                    list(self.api_call_metrics.keys()),
                    list(self.api_call_metrics.values()),
                ):
                    res = v.compute_row(item.response, item.target)
                    if k == "api_call_params":
                        out.api_call_param_names = res[0]
                        out.api_call_param_values = res[1]
                        if len(res) == 3:
                            out.api_call_param_relation = res[2]
                    else:
                        out[k] = res
                out.complete_api_call = self.complete_api_call.compute_row(
                    [out.api_call_method],
                    [(out.api_call_param_names, out.api_call_param_values)],
                )
                if item.is_multi_domain_api_call:
                    for k, v in zip(
                        list(self.multi_domain_api_call_metrics.keys()),
                        list(self.multi_domain_api_call_metrics.values()),
                    ):
                        res = v.compute_row(item.pred, item.label)
                        if "api_call_params" in k:
                            item.multi_domain_api_call_param_names = res[0]
                            item.multi_domain_api_call_param_values = res[1]
                            if len(res) == 3:
                                item.multi_domain_api_call_param_relation = res[2]
                        else:
                            item[k] = res

                    item.multi_domain_complete_api_call = (
                        self.multi_domain_complete_api_call.compute_row(
                            [item.multi_domain_api_call_method],
                            [
                                (
                                    item.multi_domain_api_call_param_names,
                                    item.multi_domain_api_call_param_values,
                                )
                            ],
                        )
                    )
                    self.multi_domain_complete_api_call.update(
                        [item.multi_domain_api_call_method],
                        [
                            (
                                item.multi_domain_api_call_param_names,
                                item.multi_domain_api_call_param_values,
                            )
                        ],
                    )
            else:
                # metrics = self.get_bleu_results(item)
                # out.response_gleu = metrics.response_gleu
                # out.response_bleu = metrics.response_bleu
                for k, v in zip(
                    list(self.response_metrics.keys()),
                    list(self.response_metrics.values()),
                ):
                    res = v.compute_row(item.response, item.target)
                    out[k] = res

            output.append(out.toDict())
        df = pd.DataFrame(output)
        df.to_csv(self.cfg.out_path, index=False, encoding="utf-8")
        a = 1


if __name__ == "__main__":
    path = "/mounts/u-amo-d1/adibm-data/projects/ZSToD/data_exploration/chatgpt/Dialogue_Responses.csv"
    out_path = "data_exploration/chatgpt/chat_gpt_all.csv"
    cgi = ChatGptInference(DotMap(path=path, out_path=out_path))
    cgi.run()
