from my_enums import Steps
from schema.pseudo_schema_dataclasses import PseudoSchema
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
)
from typing import Union
from pathlib import Path
import utils


class SchemaLoader:
    def __init__(self, schema_type: Union[DstcSchema, PseudoSchema]):
        self.schema_type = schema_type

    def get_schema_from_step(
        self, step_path: Path
    ) -> dict[str, Union[DstcSchema, PseudoSchema]]:
        schemas = {}
        path = step_path / "schema.json"
        schema_json = utils.read_json(path)
        for s in schema_json:
            schema = self.schema_type.from_dict(s)
            schema.step = step_path.name
            schemas[schema.service_name] = schema
        return schemas

    def get_schemas(
        self, base_path: Path
    ) -> dict[str, Union[DstcSchema, PseudoSchema]]:
        steps = Steps.list()
        all_schemas = {}
        for step in steps:
            step_path = base_path / step
            schema = self.get_schema_from_step(step_path)
            all_schemas.update(schema)
        return all_schemas
