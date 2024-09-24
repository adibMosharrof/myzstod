from datamodules.data_filters.base_data_filter import BaseDataFilter


class SplitPercentFilter(BaseDataFilter):
    def __init__(self, percent: float):
        self.percent = percent

    def apply(self, data: list) -> list:
        split_index = int(len(data) * self.percent)
        return data[:split_index]
