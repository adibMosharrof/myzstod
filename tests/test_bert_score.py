import sys

sys.path.append("./src")
from metrics.bert_score_metric import BertScoreMetric
from transformers import AutoTokenizer


def test_bert_score():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    bert_model = "microsoft/mpnet-base"
    scmm = BertScoreMetric(tokenizer=tokenizer, bert_score_model=bert_model)
    label = "Any preference on the restaurant, location and time?"
    pred = "Is there a particular restaurant you have in mind?"
    scmm.update([pred], [label])
    res = scmm.compute()

    label = "ApiCall(method='ReserveRestaurant', parameters='date': '2019-03-08', 'location': 'Corte Madera', 'number_of_seats': '2','restaurant_name': \"P.f. Chang's\", 'time': '12:00')"
    pred = (
        "I am trying to book 2 tables for you in the tenth of March and want to cancel."
    )
    scmm = BertScoreMetric(tokenizer=tokenizer, bert_score_model=bert_model)
    scmm.update([pred], [label])
    res = scmm.compute()
    a = 1
