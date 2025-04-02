from datamodules.tod_dataset import TodDataSet


class InterpretUtilities:
    @staticmethod
    def get_used_datasets(interpret_datasets) -> list[TodDataSet]:
        return [d for d in interpret_datasets if d.data]
