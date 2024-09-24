from abc import ABC, abstractmethod


class BaseDataFilter(ABC):
    @abstractmethod
    def apply(self, data: list) -> list:
        pass

    def raise_error_if_data_is_empty(self, data: list):
        if not len(data):
            raise ValueError("Data is empty")
