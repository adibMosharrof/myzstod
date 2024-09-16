import os
from pathlib import Path
import sys

from dotmap import DotMap

from sgd_dstc8_data_model.dstc_dataclasses import get_schemas, DstcSchema

sys.path.insert(0, os.path.abspath("./src"))

from my_enums import Steps
import numpy as np


class ApiStats:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.project_root = Path(self.cfg.project_root)

    def get_dataset_schema(self, data_path):
        schemas = {}
        for step in Steps:
            schema = get_schemas(data_path, step.value)
            schemas.update(schema)
        return schemas

    def get_api_methods(self, data_path):
        schemas = self.get_dataset_schema(data_path)
        all_intents = []
        all_slots = []
        for domain, schema in schemas.items():
            for intent in schema.intents:
                all_intents.append(intent.name)
            for slot in schema.slots:
                # all_slots.append(f"{domain}_{slot.name}")
                all_slots.append(slot.name)
        return np.unique(all_intents), np.unique(all_slots)

    def run(self):
        dataset_paths = [
            "data/dstc8-schema-guided-dialogue",
            "data/ketod",
            "data/bitod",
        ]
        results = []
        headers = ["dataset", "methods", "slots"]
        for dp in dataset_paths:
            data_path = self.cfg.project_root / dp
            methods, slots = self.get_api_methods(data_path)
            results.append([dp, len(methods), len(slots)])
        a = 1


if __name__ == "__main__":
    astats = ApiStats(
        DotMap(
            raw_data_root="data/dstc8-schema-guided-dialogue/",
            processed_data_root="data/processed_data/",
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            # data_path="data/dstc8-schema-guided-dialogue",
            data_path="data/bitod",
            out_file_path=Path("data_exploration/api_stats/api_stats.csv"),
        )
    )
    astats.run()
