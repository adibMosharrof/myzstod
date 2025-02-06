from torch.utils.data import Dataset
from pathlib import Path

from tod.turns.turn_csv_row_base import TurnCsvRowBase


class TodDataSet(Dataset):
    def __init__(
        self,
        data: list[TurnCsvRowBase],
        dataset_name: str = "",
        domain_setting: list[str] = None,
        step_name: str = "",
        raw_data_root: Path = None,
    ):
        self.data: list[TurnCsvRowBase] = data
        self.dataset_name = dataset_name
        self.domain_setting = domain_setting
        self.step_name = step_name
        self.raw_data_root = raw_data_root

    def get_domain_names(self):
        return ",".join(self.domain_setting)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> TurnCsvRowBase:
        return self.data[idx]
