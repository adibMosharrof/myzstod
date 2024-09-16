from pathlib import Path
from dotmap import DotMap
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))

from my_enums import Steps
import utils


class SchemaExtractor:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def run(self):
        steps = [Steps.TRAIN.value, Steps.DEV.value]
        all_schemas = []
        for root in self.cfg.raw_data_root:
            for step in steps:
                schema_path = self.cfg.project_root / root / step / "schema.json"
                schema_json = utils.read_json(schema_path)
                filtered_schema = [
                    schema
                    for schema in schema_json
                    if self.cfg.domain in schema["service_name"]
                ]
                all_schemas.append(filtered_schema[0])
        out_root = self.cfg.out_path / self.cfg.domain
        out_root.mkdir(parents=True, exist_ok=True)
        for schema in all_schemas:
            path = out_root / f"{schema['service_name']}_schema.json"
            utils.write_json(path=path, data=schema)


if __name__ == "__main__":
    astats = SchemaExtractor(
        DotMap(
            raw_data_root=[
                "data/dstc8-schema-guided-dialogue/",
                "data/dstc8-schema-guided-dialogue/sgd_x/data/v1",
                "data/dstc8-schema-guided-dialogue/sgd_x/data/v2",
                "data/dstc8-schema-guided-dialogue/sgd_x/data/v3",
                "data/dstc8-schema-guided-dialogue/sgd_x/data/v4",
                "data/dstc8-schema-guided-dialogue/sgd_x/data/v5",
            ],
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            out_path=Path("data_exploration/schema_extractor"),
            domain="Restaurants",
        )
    )
    astats.run()
