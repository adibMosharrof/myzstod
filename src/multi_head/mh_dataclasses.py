from dataclasses import dataclass, fields
from dataclasses_json import dataclass_json
from torch import nn
from typing import Union


@dataclass_json
@dataclass
class MultiHeadDict:
    intents: Union[nn.Linear, str, int]
    beliefs: Union[nn.Linear, str, int]
    # requested_slots: Union[nn.Linear, str, int]
    # system_actions: Union[nn.Linear, str, int]
    # user_actions: Union[nn.Linear, str, int]
    # nlg: Union[nn.Linear, str, int]

    @classmethod
    def head_names(self):
        return [field.name for field in fields(self)]

    def get_values(self):
        return [getattr(self, field.name) for field in fields(self)]

    def __getitem__(self, item):
        return getattr(self, item)
