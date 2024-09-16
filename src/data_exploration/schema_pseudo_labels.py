from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath("./src"))

from dotmap import DotMap
from sgd_dstc8_data_model.dstc_dataclasses import (
    get_schemas,
    DstcSchema,
    DstcSchemaSlot,
    DstcSchemaIntent,
)

from my_enums import Steps


class SchemaPseudoLabels:
    def __init__(self, cfg):
        self.cfg = cfg
        self.slot_counter = 1
        self.intent_counter = 1

    def get_pseudo_schema(self, schema, version_num):
        slot_map = {}
        for slot in schema.slots:
            slot_name = f"slot_{self.slot_counter}"
            self.slot_counter += 1
            dstc_slot = DstcSchemaSlot(
                name=slot_name,
                description="",
                is_categorical=slot.is_categorical,
                possible_values=[],
            )
            slot_map[slot.name] = dstc_slot
        new_intents = []
        for intent in schema.intents:
            intent_name = f"intent_{self.intent_counter}"
            self.intent_counter += 1
            dstc_intent = DstcSchemaIntent(
                name=intent_name,
                description="",
                is_transactional=intent.is_transactional,
                required_slots=[slot_map[slot].name for slot in intent.required_slots],
                optional_slots=[slot_map[slot].name for slot in intent.optional_slots],
                result_slots=[slot_map[slot].name for slot in intent.result_slots],
            )
            new_intents.append(dstc_intent)
        service_name = f"{schema.service_name.split('_')[0]}_{version_num}"
        new_schema = DstcSchema(
            service_name=service_name,
            description=schema.description,
            intents=new_intents,
            slots=list(slot_map.values()),
        )
        return new_schema
        return json.dumps(asdict(new_schema), indent=4)

    def run(self):
        data_path = self.cfg.project_root / self.cfg.data_path
        for version in range(1, self.cfg.num_versions + 1):
            version_num = f"{self.cfg.version_suffix}{version}"
            version_path = self.cfg.project_root / self.cfg.sgd_x_path / version_num
            version_path.mkdir(parents=True, exist_ok=True)
            for step in Steps:

                schemas = get_schemas(data_path, step.value)
                pseudo_schemas = [
                    self.get_pseudo_schema(schema, version_num)
                    for schema in schemas.values()
                ]
                step_dir = version_path / step.value
                step_dir.mkdir(parents=True, exist_ok=True)
                with open(step_dir / "schema.json", "w") as json_file:
                    json.dump(
                        [asdict(schema) for schema in pseudo_schemas],
                        json_file,
                        indent=4,
                    )


if __name__ == "__main__":
    spl = SchemaPseudoLabels(
        DotMap(
            processed_data_root="data/processed_data/",
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            data_path="data/dstc8-schema-guided-dialogue",
            sgd_x_path="data/dstc8-schema-guided-dialogue/sgd_x/data",
            out_file_path=Path("data_exploration/api_stats/api_stats.csv"),
            version_suffix="pl",
            num_versions=5,
        )
    )
    spl.run()
