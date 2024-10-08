from my_enums import ContextType
from schema.pseudo_schema_dataclasses import PseudoSchema
from schema.schema_loader import SchemaLoader

from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
)
from utilities.context_manager import ContextManager


class SchemaFactory:
    @staticmethod
    def create_schema_loader(context_type: str) -> SchemaLoader:
        if ContextManager.is_sgd_pseudo_labels(context_type):
            return SchemaLoader(PseudoSchema)
        else:
            return SchemaLoader(DstcSchema)
