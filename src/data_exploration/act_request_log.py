from collections import defaultdict
from dataclasses import dataclass
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

from sgd_dstc8_data_model.dstc_dataclasses import DstcDialog, DstcSchema
from my_enums import DstcSystemActions, Steps
import utils


@dataclass
class RequestLog:
    dialog_id: str
    turn_id: int
    total_turns: int


class ActReqLog:
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

    def request_log(self, dialog: DstcDialog, out: list):
        for i, turn in enumerate(dialog.turns):
            if turn.speaker == "USER":
                continue
            found = False
            for frame in turn.frames:
                for act in frame.actions:
                    if act.act == DstcSystemActions.REQUEST:
                        out.append(
                            RequestLog(
                                dialog_id=dialog.dialogue_id,
                                turn_id=i,
                                total_turns=len(dialog.turns),
                            )
                        )
                        found = True
                        break
                if found:
                    break

    def api_call_log(self, dialogs: list[DstcDialog], request_log: list[RequestLog]):
        api_call_log = defaultdict(list)
        for dial in dialogs:
            for i, turn in enumerate(dial.turns):
                for frame in turn.frames:
                    if frame.service_call:
                        api_call_log[dial.dialogue_id].append(i)
        group_data = (
            pd.DataFrame(request_log)
            .groupby("dialog_id")
            .agg({"turn_id": np.max, "total_turns": np.max})
        )
        a = 1

    def dialogs_wo_request(self, dialogs, data):
        df = pd.DataFrame(data)
        req_log_dial_ids = set(df["dialog_id"])
        all_dials = []
        for k, v in dialogs.items():
            for _, dials in v.items():
                for d in dials:
                    all_dials.append(d.dialogue_id)
        unique_dials = set(all_dials)
        dials_wo_req = unique_dials - req_log_dial_ids
        out = {}
        out["without_act_request_ratio"] = round(
            len(dials_wo_req) / len(unique_dials), 3
        )

        out_file_path = self.cfg.out_file_path.parent / "dials_wo_req.json"
        out_path = (
            str(out_file_path) + "_".join(map(str, self.cfg.num_dialogs)) + ".json"
        )
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(json.dumps(out, default=str))

    def turn_stats(self, data):
        df = pd.DataFrame(data)
        out = {}
        group_data = df.groupby("dialog_id").agg(
            {"turn_id": np.max, "total_turns": np.max}
        )
        out["ratio"] = (group_data["turn_id"] / group_data["total_turns"]).mean()
        out["turn_avg"] = group_data["turn_id"].mean()
        out["turn_median"] = group_data["turn_id"].median()
        out["turn_max"] = group_data["turn_id"].max()
        out["turn_min"] = group_data["turn_id"].min()
        out["total_turns_avg"] = group_data["total_turns"].mean()
        out["total_turns_median"] = group_data["total_turns"].median()
        out["total_turns_max"] = group_data["total_turns"].max()
        out["total_turns_min"] = group_data["total_turns"].min()

        out_path = (
            str(self.cfg.out_file_path)
            + "_".join(map(str, self.cfg.num_dialogs))
            + ".json"
        )
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(json.dumps(out, default=str))

    def run(self):
        dialogs = self._get_dialogs()
        all_dialogs = []
        for k, v in dialogs.items():
            for _, dials in v.items():
                for d in dials:
                    all_dialogs.append(d)
        request_log = []
        for dial in all_dialogs:
            self.request_log(dial, request_log)
        # self.turn_stats(out)
        self.dialogs_wo_request(dialogs, request_log)
        # self.api_call_log(all_dialogs, request_log)
        a = 1


if __name__ == "__main__":
    dd = ActReqLog(
        DotMap(
            raw_data_root="data/dstc8-schema-guided-dialogue/",
            processed_data_root="data/processed_data/",
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            num_dialogs=[127, 20, 34],
            # num_dialogs=[1, 1, 1],
            out_file_path=Path("data_exploration/statistics/act_request_log"),
        )
    )
    dd.run()
