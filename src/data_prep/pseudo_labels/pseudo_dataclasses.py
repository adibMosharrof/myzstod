from dataclasses import dataclass
from sgd_dstc8_data_model.dstc_dataclasses import DstcSchemaSlot, DstcSchemaIntent


@dataclass
class PseudoSchemaSlot(DstcSchemaSlot):
    pseudo_name: str


@dataclass
class PseudoSchemaIntent(DstcSchemaIntent):
    pseudo_name: str
