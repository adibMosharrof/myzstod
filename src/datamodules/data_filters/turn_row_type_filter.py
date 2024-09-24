from datamodules.data_filters.base_data_filter import BaseDataFilter


class TurnRowTypeFilter(BaseDataFilter):
    def __init__(self, turn_row_type: int):
        self.turn_row_type = turn_row_type

    def apply(self, data: list) -> list:
        # self.raise_error_if_data_is_empty(data)
        if not len(data):
            return []
        if not hasattr(data[0], "turn_row_type"):
            raise ValueError("Data does not have 'turn_row_type' field")

        return [row for row in data if row.turn_row_type == self.turn_row_type]
