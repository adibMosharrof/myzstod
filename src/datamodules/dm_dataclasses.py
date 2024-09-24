from dataclasses import dataclass
from typing import Union

from my_enums import Steps


@dataclass(frozen=True)
class StepData:
    name: Steps
    num_dialog: int
    overwrite: bool
    split_percent: float
    domain_settings: Union[list[str], str]
