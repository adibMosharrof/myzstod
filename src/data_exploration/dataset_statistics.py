import json
import os
from pathlib import Path
import sys
from dotmap import DotMap
import numpy as np
import pandas as pd
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
)

from datasets import load_dataset


sys.path.insert(0, os.path.abspath("./src"))
sys.path.insert(0, os.path.abspath("./"))

from my_enums import Steps
from schema.schema_loader import SchemaLoader

from utilities.dialog_studio_dataclasses import DsDialog
import data_prep.data_prep_utils as data_prep_utils


class DatasetStatistics:
    def __init__(self, cfg):
        self.cfg = cfg

    def init_variables(self):
        self.processed_domains = []
        self.train_intents = []
        self.train_slots = []
        self.test_intents = []
        self.test_slots = []
        self.dev_intents = []
        self.dev_slots = []

    def get_schema_map(self, all_schemas):
        train = list(all_schemas["train"].keys())
        test = list(all_schemas["test"].keys())

        # Step 2: Match all occurrences from test to train
        matching_items = {}
        for t_item in test:
            base = t_item.split("_")[0]
            matches = [
                train_item for train_item in train if train_item.split("_")[0] == base
            ]
            if matches:
                matching_items[t_item] = matches
        return matching_items

    def add_domain_statistics(self, domain, schema: DstcSchema, intents, slots):
        # if domain in self.processed_domains:
        #     return
        domain_name = domain.split("_")[0]
        # intent_names = [s.name for s in schema.intents]
        # slot_names = [ s.name for s in schema.slots]
        # intent_names = ["_".join([domain_name, s.name]) for s in schema.intents]
        # slot_names = ["_".join([domain_name, s.name]) for s in schema.slots]
        intent_names = [s.name for s in schema.intents]
        slot_names = [s.name for s in schema.slots]
        intents.extend(intent_names)
        slots.extend(slot_names)
        # self.processed_domains.append(domain)

    def get_dataset_statistics(self, dataset_name, all_schemas):
        self.init_variables()
        # schema_loader = SchemaLoader(DstcSchema)
        # steps = Steps.list()
        # all_schemas = {}

        # for step in steps:
        #     step_path = dataset.raw_data_root / step
        #     all_schemas[step] = schema_loader.get_schema_from_step(step_path)
        # schema_map = self.get_schema_map(all_schemas)
        for test_domain, test_schema in all_schemas["test"].items():
            self.add_domain_statistics(
                test_domain,
                test_schema,
                self.test_intents,
                self.test_slots,
            )
        for train_domain, train_schema in all_schemas["train"].items():
            self.add_domain_statistics(
                train_domain,
                train_schema,
                self.train_intents,
                self.train_slots,
            )
        for dev_domain, dev_schema in all_schemas["dev"].items():
            self.add_domain_statistics(
                dev_domain,
                dev_schema,
                self.dev_intents,
                self.dev_slots,
            )
        u_train_intents = np.unique(self.train_intents)
        u_train_slots = np.unique(self.train_slots)
        u_test_intents = np.unique(self.test_intents)
        u_test_slots = np.unique(self.test_slots)
        u_dev_intents = np.unique(self.dev_intents)
        u_dev_slots = np.unique(self.dev_slots)

        # 1. Total unique intents and slots in both train and test
        all_intents = np.union1d(
            np.union1d(u_train_intents, u_dev_intents), u_test_intents
        )
        all_slots = np.union1d(np.union1d(u_train_slots, u_dev_slots), u_test_slots)

        # 2. Unique intents and slots in test but not in train
        unseen_intents = np.setdiff1d(u_test_intents, u_train_intents)
        unseen_slots = np.setdiff1d(u_test_slots, u_train_slots)

        return DotMap(
            dataset=dataset_name,
            all_intents=len(all_intents),
            all_slots=len(all_slots),
            unseen_intents=len(unseen_intents),
            unseen_slots=len(unseen_slots),
        )

    def get_sgd_schema(self, dataset):
        schema_loader = SchemaLoader(DstcSchema)
        steps = Steps.list()
        schemas = {}

        for step in steps:
            step_path = dataset.raw_data_root / step
            schemas[step] = schema_loader.get_schema_from_step(step_path)
        return schemas

    def get_ketod_schema(self, sgd_dataset):
        schema_loader = SchemaLoader(DstcSchema)
        all_schemas = schema_loader.get_schemas(sgd_dataset.raw_data_root)
        steps = Steps.list()
        all_domains = {}
        schemas = {}
        p = "data_exploration/data_statistics/ketod_domains.json"
        if Path(p).exists():
            with open(p, "r") as f:
                schemas = json.load(f)
            out = {}
            for step_name in steps:
                out[step_name] = {}
                for domain in schemas[step_name]:
                    out[step_name][domain] = all_schemas[domain]
            return out
        dataset = load_dataset("Salesforce/dialogstudio", "KETOD")
        for step_name in steps:
            ds = data_prep_utils.get_dialog_studio_step_data(step_name, dataset)
            step_domains = []
            for row in ds:
                dialog = DsDialog(row)
                metadata = json.loads(dialog.external_knowledge_non_flat)
                dialog_domains = list(metadata["metadata"].keys())
                step_domains.extend(dialog_domains)
            domains = np.unique(step_domains)
            all_domains[step_name] = list(domains)
            schemas[step_name] = {d: all_schemas[d] for d in domains}
        with open(p, "w") as f:
            json.dump(all_domains, f)
        return schemas

    def run(self):
        out = []
        # for dataset in self.cfg.datasets:
        # out.append(self.get_dataset_statistics(dataset))
        ketod_schemas = self.get_ketod_schema(self.cfg.datasets)
        sgd_schemas = self.get_sgd_schema(self.cfg.datasets)
        for dataset_name, schemas in zip(
            ["KETOD", "SGD"], [ketod_schemas, sgd_schemas]
        ):
            out.append(dict(self.get_dataset_statistics(dataset_name, schemas)))
        df = pd.DataFrame(out)
        df.to_csv(self.cfg.out_path, index=False)
        a = 1


if __name__ == "__main__":
    ds = DatasetStatistics(
        DotMap(
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            datasets=DotMap(
                name="sgd",
                raw_data_root=Path("data/dstc8-schema-guided-dialogue"),
            ),
            out_path=Path("data_exploration/data_statistics/data_statistics.csv"),
        )
    )
    ds.run()
