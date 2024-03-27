import sys
import os


sys.path.append("./src")
from metrics.bitod_api_call_parameters_metric import BitodApiCallParametersMetric


def test_bitod_apicall_extract_params():
    metric = BitodApiCallParametersMetric()
    input_str = "ApiCall(method=hotels_booking, parameters={name equal_to Cordis, Hong Kong|start_day equal_to 23|start_month equal_to 11|number_of_nights equal_to 9|user_name equal_to David|number_of_rooms equal_to eight})"
    params = metric._get_parameters_from_text(input_str)
    assert len(params) == 6


def test_bitod_apicall_params():
    metric = BitodApiCallParametersMetric()
    ref_str = [
        "ApiCall(method=hotels_search, parameters=rating at_least four|stars equal_to don't care|location equal_to don't care|price_level equal_to expensive)",
        "ApiCall(method=hotels_booking, parameters=name equal_to Cordis, Hong Kong|start_day equal_to 23|start_month equal_to 11|number_of_nights equal_to 9|user_name equal_to David|number_of_rooms equal_to eight)",
    ]
    pred_str = [
        "What can I do for you?",
        "ApiCall(method=hotels_booking, parameters=name equal_to Cordis, Hong Kong|start_day equal_to 23|start_month equal_to 11|number_of_nights equal_to 9|user_name equal_to David|number_of_rooms equal_to eight)",
    ]
    metric.update(
        references=ref_str,
        predictions=pred_str,
    )
    out = metric.compute()
    assert out[0] == 1.0
    assert out[1] == 1.0
    assert out[2] == 1.0
