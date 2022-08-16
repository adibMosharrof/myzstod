import pytest

import os
import sys

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(myPath + "/../src"))
from tod_metrics import RequestedSlotsMetric
import math


@pytest.fixture
def metric():
    return RequestedSlotsMetric()


class TestRequestedSlot:
    def base(self, metric, preds, refs, expected):
        metric.add_batch(preds, refs)
        f1 = metric.compute()
        assert round(f1, 2) == expected

    @pytest.mark.parametrize(
        "preds, refs, expected",
        [
            (
                [
                    """<|begintarget|><|beginintent|>FindMovies<|endintent|>
                    <|beginbelief|><|endbelief|>
                    <|beginaction|>REQUEST->Media_genre<-<|endaction|>
                    <|beginresponse|>What kind of movies are you interested in?<|endresponse|>""",
                    """<|beginintent|>NONE<|endintent|>
                    <|beginbelief|>Media_genre->ghost|Media_title->Midsommar<|endbelief|>
                    <|beginaction|>REQ_MORE->Media_<-<|endaction|>
                    <|beginresponse|>Is there anything else I can do for you?<|endresponse|>""",
                ],
                [
                    """<|beginintent|>FindMovies<|endintent|>
                    <|beginbelief|><|endbelief|>
                    <|beginaction|>REQUEST->Media_genre<-<|endaction|>
                    <|beginresponse|>Sure, what kind of movies do you like?<|endresponse|>""",
                    """<|begintarget|><|beginintent|>FindAttractions<|endintent|>
                    <|beginbelief|>Travel_location->Paris, France<|endbelief|>
                    <|beginaction|>OFFER->Travel_attractionName<-American Church in Paris|OFFER->Travel_category<-Place of Worship<|endaction|>
                    <|beginresponse|>How about American Church in Paris? It's a place of worship.<|endresponse|>""",
                ],
                0.0,
            )
        ],
    )
    def test_no_requested_slots(self, metric, preds, refs, expected):
        return self.base(metric, preds, refs, expected)

    @pytest.mark.parametrize(
        "preds, refs, expected",
        [
            (
                [
                    """<|beginrequestedslots|>Travel_freeEntry|Travel_phoneNumber<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_year<|endrequestedslots|>""",
                ],
                [
                    """<|beginrequestedslots|>Travel_freeEntry|Travel_phoneNumber<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_year<|endrequestedslots|>""",
                ],
                100.0,
            )
        ],
    )
    def test_all_correct(self, metric, preds, refs, expected):
        return self.base(metric, preds, refs, expected)

    @pytest.mark.parametrize(
        "preds, refs, expected",
        [
            (
                [
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_year<|endrequestedslots|>""",
                ],
                [
                    """<|beginrequestedslots|>Media_artist<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_directedBy<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_genre<|endrequestedslots|>""",
                ],
                0.0,
            )
        ],
    )
    def test_all_incorrect(self, metric, preds, refs, expected):
        return self.base(metric, preds, refs, expected)

    @pytest.mark.parametrize(
        "preds, refs, expected",
        [
            (
                [
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_directedBy<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_year<|endrequestedslots|>""",
                ],
                [
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_year<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_directedBy<|endrequestedslots|>""",
                ],
                33.33,
            ),
            (
                [
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_directedBy<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_year<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_directedBy<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_year<|endrequestedslots|>""",
                ],
                [
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Music_year<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_directedBy<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_starring<|endrequestedslots|>""",
                    """<|beginrequestedslots|>Media_directedBy<|endrequestedslots|>""",
                ],
                33.33,
                # 26.67 for macro f1 score
            ),
        ],
    )
    def test_diff_values(self, metric, preds, refs, expected):
        return self.base(metric, preds, refs, expected)
