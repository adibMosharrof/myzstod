import sys

sys.path.append("./src")
from metrics.complete_api_call_metric import CompleteApiCallMetric

from torchmetrics import MetricCollection
from transformers import AutoTokenizer
import pandas as pd
from metrics.api_call_method_metric import ApiCallMethodMetric
from metrics.api_call_parameters_metric import ApiCallParametersMetric
from metrics.bert_score_metric import BertScoreMetric

from metrics.nlg_gleu_metric import NlgGleuMetric


csv_path = "playground/data/nlg_api_call_preds.csv"


def test_row_wise_gleu_score():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    df = pd.read_csv(csv_path)
    metric_coll = MetricCollection(
        {
            "response_gleu": NlgGleuMetric(tokenizer),
            "response_bertscore": BertScoreMetric(tokenizer),
            "api_call_method": ApiCallMethodMetric(),
            "api_call_params_metric": ApiCallParametersMetric(),
        }
    )
    complete_api_call_metric = CompleteApiCallMetric()
    complete_api_call = "complete_api_call"
    for i, row in df.iterrows():
        for k, v in metric_coll.items():
            res = v.compute_row(row["pred"], row["label"])
            row[k] = res
        row[complete_api_call] = complete_api_call_metric.update(
            row["api_call_method"], row["api_call_params_metric"]
        )
        a = 1
    # a = 1
