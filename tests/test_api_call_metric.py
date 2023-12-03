import sys
import os


sys.path.append("./src")

from metrics.api_call_parameters_metric import ServiceCallParametersMetric
from metrics.api_call_method_metric import ApiCallMethodMetric


def test_method_extract():
    scmm = ApiCallMethodMetric()
    input_string = "ServiceCall(method='GetTrainTickets', parameters='from': 'San Diego', 'leaving': '2019-03-14', 'journey_start_time': '11:20', 'number_of_adults': '1', 'trip_protection': 'True', 'origin': 'Phoenix')"
    res = scmm._get_method_from_text(input_string)
    assert res == "GetTrainTickets"
    input_string = "ServiceCall(method='', parameters='from': 'San Diego', 'leaving': '2019-03-14', 'journey_start_time': '11:20', 'number_of_adults': '1', 'trip_protection': 'True', 'origin': 'Phoenix')"
    res = scmm._get_method_from_text(input_string)
    assert res == ""


def test_parameters_extract():
    input_string = "ServiceCall(method='GetTrainTickets', parameters='from': 'San Diego', 'leaving': '2019-03-14')"
    scpm = ServiceCallParametersMetric()
    res = scpm._get_parameters_from_text(input_string)
    assert res == {"from": "San Diego", "leaving": "2019-03-14"}
