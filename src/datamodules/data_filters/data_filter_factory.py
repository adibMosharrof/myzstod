from enum import Enum

from dotmap import DotMap
from datamodules.data_filters.api_in_context_filter import ApiInContextFilter
from datamodules.data_filters.turn_row_type_filter import TurnRowTypeFilter


class FilterNames(str, Enum):
    API_CALL = "api_call"
    API_IN_CONTEXT = "api_in_context"


class DataFilterFactory:
    @staticmethod
    def get_data_filter(
        name: str,
        cfg: DotMap,
        collator=None,
    ):
        if name == FilterNames.API_CALL.value:
            return TurnRowTypeFilter(turn_row_type=1)
        if name == FilterNames.API_IN_CONTEXT.value:
            return ApiInContextFilter(collator=collator)
        raise ValueError(f"Unknown data filter: {name}")
