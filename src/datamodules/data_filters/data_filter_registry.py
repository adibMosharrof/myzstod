from datamodules.data_filters.turn_row_type_filter import TurnRowTypeFilter


DATA_FILTER_MAP = {
    "api_call": TurnRowTypeFilter(turn_row_type=1),
}
