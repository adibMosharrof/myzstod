import csv
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from dotmap import DotMap
import pandas as pd
import numpy as np


class Speaker(str, Enum):
    USER = "user"
    SYS = "system"
    SEARCH = "search_results"


@dataclass
class UserStudyOutput:
    dialog_id: str
    domains: str
    dialog_history: str
    ground_truth_dialog_history: str


@dataclass
class Turn:
    speaker: str
    utterance: str

    def __str__(self):
        return f"{self.speaker}: {self.utterance}"


@dataclass
class Dialog:
    dialog_id: str
    turns: list
    services: list[str]

    def get_dialog_history(self):
        return "\n".join([str(turn) for turn in self.turns])

    def get_user_study_output(self):
        return UserStudyOutput(
            dialog_id=self.dialog_id,
            domains=",".join(self.services),
            dialog_history=self.get_dialog_history(),
            ground_truth_dialog_history=self.get_ground_truth_dialog_history(),
        )


class UserStudy:
    def __init__(self, cfg):
        self.cfg = cfg

    def write_to_csv(self, data):

        out_path = (
            self.cfg.project_root
            / self.cfg.out_root
            / self.cfg.dataset_name
            / f"{self.cfg.model_name}_user_study.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        field_names = [f.name for f in fields(UserStudyOutput)]
        with open(out_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for d in data:
                writer.writerow(asdict(d))

    def read_data(self, model_names):
        data_path = self.cfg.project_root / self.cfg.data_path / self.cfg.dataset_name
        ids_path = data_path / "ids.txt"
        with open(ids_path, "r") as f:
            ids = [int(line.strip()) for line in f]
        # ids = ids[:5]
        data = DotMap()
        for model_name in model_names:
            pred_path = data_path / f"{model_name}_{self.cfg.dataset_name}_all.csv"
            pred_df = pd.read_csv(pred_path)
            data[model_name] = pred_df[pred_df.dialog_id.isin(ids)]
        return data

    def get_dialog_history(self, data, column_name):
        grouped = data.sort_values(by=["dialog_id", "turn_id"]).groupby("dialog_id")
        out = []
        for id, group in grouped:
            turns = []
            sys_cols = group[column_name]
            for user, sys, search_results, turn_row_type in zip(
                group.current_user_utterance,
                sys_cols,
                group.search_results,
                group.turn_row_type,
            ):
                if turn_row_type == 2:
                    continue
                turns.append(Turn(Speaker.USER, user))
                if type(search_results) == str:
                    turns.append(Turn(Speaker.SEARCH, search_results))
                turns.append(Turn(Speaker.SYS, sys))
            dialog = Dialog(id, turns, group.domains.unique())
            out.append(dialog.get_dialog_history())
        return out

    def get_domains_and_ids(self, data):
        grouped = data.sort_values(by=["dialog_id", "turn_id"]).groupby("dialog_id")
        domains = []
        ids = []
        for id, group in grouped:
            domains.append(",".join(group.domains.unique()))
            ids.append(id)
        return DotMap(
            domains=domains,
            ids=ids,
        )

    def run(self):
        model_names = ["soloist", "autotod", "gpt", "llama", "flan"]
        data = self.read_data(model_names)

        out = self.get_domains_and_ids(data.gpt)
        out["gt"] = self.get_dialog_history(data.gpt, "label")
        for model_name in model_names:
            out[model_name] = self.get_dialog_history(data[model_name], "pred")
        df = pd.DataFrame(out)
        out_path = (
            self.cfg.project_root
            / self.cfg.out_root
            / f"{self.cfg.dataset_name}_user_study.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            out_path,
            index=False,
        )

    def old_run(self):
        csv_path = self.cfg.project_root / self.cfg.pred_path
        csv_df = pd.read_csv(csv_path)
        all_ids = csv_df.dialog_id.unique()
        sample_ids = np.random.choice(all_ids, size=10, replace=False)
        sample_df = csv_df[csv_df.dialog_id.isin(sample_ids)]
        grouped = sample_df.sort_values(by=["dialog_id", "turn_id"]).groupby(
            "dialog_id"
        )
        out = []
        for id, group in grouped:
            turns = []
            for user, sys, gt_sys, search_results in zip(
                group.current_user_utterance,
                group.pred,
                group.label,
                group.search_results,
            ):
                turns.append(Turn(Speaker.USER, user))
                if type(search_results) == str:
                    turns.append(Turn(Speaker.SEARCH, search_results))
                turns.append(Turn(Speaker.SYS, sys))
                turns.append(Turn(Speaker.GT_SYS, gt_sys))
            dialog = Dialog(id, turns, group.domains.unique())
            out.append(dialog.get_user_study_output())
        self.write_to_csv(out)


if __name__ == "__main__":
    us = UserStudy(
        DotMap(
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            data_path="data/user_study",
            out_root="data_exploration/user_study",
            dataset_name="sgd",
            # dataset_name="ketod",
        )
    )
    us.run()
