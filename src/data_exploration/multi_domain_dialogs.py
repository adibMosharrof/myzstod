from collections import Counter, defaultdict
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys

from dotmap import DotMap
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("./src"))

from dstc.dstc_dataclasses import DstcDialog, DstcSchema
from my_enums import Steps
import utils

# myPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.abspath(myPath + "/../"))


class MultiDomainDialogs:
    def __init__(self, cfg):
        self.cfg = cfg

    def _read_dialog(self, file)-> tuple[str, DstcDialog]:
        json_dialogs = utils.read_json(file)
        dstc_dialogs = [DstcDialog.from_dict(js) for js in json_dialogs]
        dialog_id = file.stem.split("_")[-1]
        return dialog_id, dstc_dialogs

    def _get_dialogs(self) -> list[DstcDialog]:
        dialogs = []
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
            for _, dials in res:
                dialogs.append(dials)
        return np.concatenate(dialogs, axis=0)

    def _get_multi_domain_dialogs(self, dialogs:list[DstcDialog])-> tuple[str, int]:
        groups = []
        for dial in dialogs:
            if len(dial.services) > 1:
                groups.append(" ".join(sorted(dial.short_services)))
        return sorted(Counter(groups).items())
        

    def print_multi_domain_counts(self, groups: tuple[str, int]):
        headers = ["Domains", "Count"]
        rows = [[domain, count] for domain, count in groups]
        
        file_name = "".join(
            [self.cfg.out_file_path, "_".join(map(str, self.cfg.num_dialogs)), ".csv"]
        )
        utils.write_csv(headers, rows, self.cfg.project_root / file_name)

    def run(self):
        dialogs = self._get_dialogs()
        domain_counts = self._get_multi_domain_dialogs(dialogs)
        self.print_multi_domain_counts(domain_counts)


if __name__ == "__main__":
    mdd = MultiDomainDialogs(
        DotMap(
            raw_data_root="data/dstc8-schema-guided-dialogue/",
            processed_data_root="data/processed_data/",
            project_root=Path("/mounts/u-amo-d0/grad/adibm/projects/generative_tod/"),
            num_dialogs=[127, 20, 34],
            # num_dialogs=[1, 1, 1],
            out_file_path="data_exploration/multi_domain_dialogs",
        )
    )
    mdd.run()
