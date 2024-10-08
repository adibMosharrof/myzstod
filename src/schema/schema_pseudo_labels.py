from dataclasses import asdict
import json
import os
from pathlib import Path
import random
from schema.schema_loader import SchemaLoader
import string
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, os.path.abspath("./src"))
from data_prep.pseudo_labels.pseudo_dataclasses import (
    PseudoSchemaSlot,
    PseudoSchemaIntent,
)


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
        cfg.project_root = Path(cfg.project_root)
        self.cfg = cfg
        self.generated_intent_names = set()
        self.generated_slot_names = set()
        self.schema_loader = SchemaLoader(DstcSchema)

    def get_pseudo_schema(self, schema: DstcSchema, version_num):
        slot_map = {}
        for slot in schema.slots:
            slot_name = self._generate_unique_name(
                self.generated_slot_names, uppercase=False
            )
            dstc_slot = PseudoSchemaSlot(
                name=slot.name,
                pseudo_name=slot_name,
                description="",
                is_categorical=slot.is_categorical,
                possible_values=[],
            )
            slot_map[slot.name] = dstc_slot
        new_intents = []
        for intent in schema.intents:
            intent_name = self._generate_unique_name(
                self.generated_intent_names, uppercase=True
            )
            dstc_intent = PseudoSchemaIntent(
                name=intent.name,
                pseudo_name=intent_name,
                description="",
                is_transactional=intent.is_transactional,
                required_slots=[slot_map[slot].name for slot in intent.required_slots],
                optional_slots=[slot_map[slot].name for slot in intent.optional_slots],
                result_slots=[slot_map[slot].name for slot in intent.result_slots],
            )
            new_intents.append(dstc_intent)
        service_name = f"{schema.service_name}{version_num}"
        new_schema = DstcSchema(
            service_name=service_name,
            description=schema.description,
            intents=new_intents,
            slots=list(slot_map.values()),
        )
        return new_schema

    def _generate_random_string(self, length=3, uppercase=False):
        characters = string.ascii_uppercase if uppercase else string.ascii_lowercase
        return "".join(random.choices(characters, k=length))

    def _generate_unique_name(self, generated_names, uppercase=False):
        while True:
            name = self._generate_random_string(uppercase=uppercase)
            if name not in generated_names:
                generated_names.add(name)
                return name

    def run(self):
        data_path = self.cfg.project_root / self.cfg.data_path
        for version in range(0, self.cfg.num_versions):
            version_num = f"{self.cfg.version_suffix}{version}"
            version_path = self.cfg.project_root / self.cfg.sgd_x_path / version_num
            version_path.mkdir(parents=True, exist_ok=True)
            if version > 0:
                data_path = self.cfg.project_root / self.cfg.sgd_x_path / f"v{version}"
            for step in Steps:

                schemas = self.schema_loader.get_schema_from_step(data_path, step.value)
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


@hydra.main(
    config_path="../../../config/data_prep/", config_name="schema_pseudo_labels"
)
def hydra_start(cfg: DictConfig) -> None:
    spl = SchemaPseudoLabels(cfg)
    spl.run()


if __name__ == "__main__":
    hydra_start()
