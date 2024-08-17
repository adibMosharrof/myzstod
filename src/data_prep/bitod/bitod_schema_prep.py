from pathlib import Path
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))

import utils
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
    DstcSchemaSlot,
    DstcSchemaIntent,
)
from my_enums import Steps


class BitodSchemaPrep:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.project_root = Path(self.cfg.project_root)

    def run(self):
        dataset = load_dataset("Salesforce/dialogstudio", "BiTOD")
        otgy_path = self.cfg.project_root / self.cfg.raw_data_root / self.cfg.otgy_file
        otgy = utils.read_json(otgy_path)
        schemas = []
        for domain_name, item in otgy.items():
            slots = []
            for slot_name in item["slots"].keys():
                slot = DstcSchemaSlot(
                    name=slot_name,
                    description="",
                    is_categorical=False,
                    possible_values=[],
                )
                slots.append(slot)
            intents = [
                DstcSchemaIntent(
                    name=i,
                    description="",
                    is_transactional=False,
                    required_slots=[],
                    optional_slots=[],
                    result_slots=[],
                )
                for i in item["intents"]
            ]
            schema = DstcSchema(
                service_name=domain_name,
                description="",
                intents=intents,
                slots=slots,
            )
            schemas.append(schema.to_dict())
        for step_name in Steps.list():
            out_path = (
                self.cfg.project_root
                / self.cfg.raw_data_root
                / step_name
                / self.cfg.out_file
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            utils.write_json(schemas, out_path)
        a = 1


@hydra.main(config_path="../../../config/data_prep/", config_name="bitod_schema_prep")
def hydra_start(cfg: DictConfig) -> None:
    btsp = BitodSchemaPrep(cfg)
    btsp.run()


if __name__ == "__main__":
    hydra_start()
