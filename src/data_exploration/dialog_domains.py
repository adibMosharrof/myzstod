from collections import defaultdict
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys

from dotmap import DotMap
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("./src"))

from dstc.dstc_dataclasses import DstcDialog, DstcSchema
from my_enums import Steps
import utils

# myPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.abspath(myPath + "/../"))


class DialogDomains:
    def __init__(self, cfg):
        self.cfg = cfg

    def _read_dialog(self, file):
        json_dialogs = utils.read_json(file)
        dstc_dialogs = [DstcDialog.from_dict(js) for js in json_dialogs]
        dialog_id = file.stem.split("_")[-1]
        return dialog_id, dstc_dialogs

    def _get_dialogs(self):
        dialogs_by_steps = {step.value: defaultdict(dict) for step in Steps}
        for step, num_dialog in zip(Steps, self.cfg.num_dialogs):
            files = [
                file
                for file in (
                    self.cfg.project_root / self.cfg.raw_data_root / step.value
                ).iterdir()
                if not "schema.json" in file.name
            ][:num_dialog]
            res = list(
                tqdm(
                    Pool().imap(
                        self._read_dialog,
                        files,
                    ),
                    total=len(files),
                )
            )
            for id, dials in res:
                dialogs_by_steps[step.value][id] = dials
        return dialogs_by_steps

    def _group_dialogs_by_domain(self, dialogs_by_steps):
        domains = {step.value: defaultdict(dict) for step in Steps}
        for step in Steps:
            schemas: list[DstcSchema] = [
                DstcSchema.from_dict(schema_dict)
                for schema_dict in utils.read_json(
                    self.cfg.project_root
                    / self.cfg.raw_data_root
                    / step.value
                    / "schema.json"
                )
            ]

            for schema in schemas:
                domains[step.value][schema.service_name] = set()
            for file_id, dialogs in dialogs_by_steps[step.value].items():
                for dialog in dialogs:
                    for service in dialog.services:
                        domains[step.value][service].add(file_id)
        return domains

    def print_dialog_domains(self, domains):
        headers = ["Step", "Domain", "Num Dialogs", "File Ids"]
        rows = []
        self.cfg.out_file_path.mkdir(parents=True, exist_ok=True)
        for step in Steps:
            for domain, file_ids in domains[step.value].items():
                rows.append(
                    [step.value, domain, len(file_ids), " ".join(sorted(file_ids))]
                )
        file_name = "".join(
            [str(self.cfg.out_file_path), "_".join(map(str, self.cfg.num_dialogs)), ".csv"]
        )
        utils.write_csv(headers, rows, self.cfg.project_root / file_name)


    def domain_counts(self, dialogs_by_steps):
        domains = {step.value: defaultdict(lambda: 0) for step in Steps}
        for step in Steps:
            for _, dialogs in dialogs_by_steps[step.value].items():
                for dialog in dialogs:
                    service_str = ",".join(sorted(dialog.services))
                    domains[step.value][service_str] += 1
        
        headers = ["Step", "Domain", "Count"]
        rows = []
        for step in Steps:
            for domain, count in domains[step.value].items():
                rows.append([step.value, domain, count])
        
        df = pd.DataFrame(rows, columns=headers)
        df.sort_values(by=["Step", "Domain"], inplace=True)        

        file_name = "".join(
            [str(self.cfg.out_file_path), "_".join(map(str, self.cfg.num_dialogs)), ".csv"]
        )
        df.to_csv(self.cfg.project_root / file_name, index=False)
        # utils.write_csv(headers, rows, self.cfg.project_root / file_name)        


    def run(self):
        dialogs = self._get_dialogs()
        # domains = self._group_dialogs_by_domain(dialogs)
        # self.print_dialog_domains(domains)
        self.domain_counts(dialogs)
        a = 1


if __name__ == "__main__":
    dd = DialogDomains(
        DotMap(
            raw_data_root="data/dstc8-schema-guided-dialogue/",
            processed_data_root="data/processed_data/",
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            num_dialogs=[127, 20, 34],
            # num_dialogs=[1, 1, 1],
            out_file_path=Path("data_exploration/dialog_domains/domain_counts"),
        )
    )
    dd.run()
