import sys
import os

sys.path.append("./src")

from utilities import text_utilities


def test_nlg_service_name():
    service_name = "Hotels_2"
    nlg_service_name = text_utilities.get_nlg_service_name(service_name)
    assert "hotels" == nlg_service_name
    service_name = "RideSharing_1"
    nlg_service_name = text_utilities.get_nlg_service_name(service_name)
    assert "ride sharing" == nlg_service_name
