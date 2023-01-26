from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from configs.data_model_exploration_config import DataModelExplorationConfig
from configs.dm_config import DataModuleConfig

import dstc.dstc_utils as dstc_utils
from my_enums import Steps
from tod_datamodules import TodDataModule
from simple_tod_dataclasses import (
    TodTurnCsvRow,
)
from utils import read_csv_dataclass


class DataModelExploration:
    def __init__(self, cfg: DataModelExplorationConfig):
        self.cfg = cfg
        self.dm = TodDataModule(DataModuleConfig.from_data_model_exploration(cfg))

    def _get_simple_tod_rows(self) -> list[TodTurnCsvRow]:
        steps = Steps.list()
        rows = []
        for step, num_dialog in tqdm(zip(steps, self.cfg.num_dialogs)):
            csv_file_path = dstc_utils.get_csv_data_path(
                step,
                num_dialog,
                cfg=self.cfg,
            )
            rows.append(read_csv_dataclass(csv_file_path, TodTurnCsvRow))
        return np.concatenate(rows, axis=0)

    def plot_model_size(self):
        rows = self._get_simple_tod_rows()
        x_axis = np.arange(1, self.num_turns)
        turns = {
            "token_freq": [],
            "turn_max_len": {i: 0 for i in range(1, self.num_turns)},
            "turn_avg_len": {i: [] for i in range(1, self.num_turns)},
            "turn_context_max_len": {i: 0 for i in range(1, self.num_turns)},
            "turn_context_avg_len": {i: [] for i in range(1, self.num_turns)},
            "turn_target_max_len": {i: 0 for i in range(1, self.num_turns)},
            "turn_target_avg_len": {i: [] for i in range(1, self.num_turns)},
        }
        for row in tqdm(rows):
            c = self.tokenizer(row.context, return_tensors="pt")
            t = self.tokenizer(row.target, return_tensors="pt")
            context_len = c.data["input_ids"].shape[1]
            target_len = t.data["input_ids"].shape[1]
            text_len = context_len + target_len

            turns["token_freq"].append(text_len)
            turn_id = int(row.turn_id)
            turns["turn_avg_len"][turn_id].append(text_len)
            if turns["turn_max_len"][turn_id] < text_len:
                turns["turn_max_len"][turn_id] = text_len

            turns["turn_context_avg_len"][turn_id].append(context_len)
            turns["turn_target_avg_len"][turn_id].append(target_len)
            if turns["turn_context_max_len"][turn_id] < context_len:
                turns["turn_context_max_len"][turn_id] = context_len
            if turns["turn_target_max_len"][turn_id] < target_len:
                turns["turn_target_max_len"][turn_id] = target_len

        plt.style.use("ggplot")

        fig1 = plt.hist(turns["token_freq"], bins=20)
        plt.xlabel("Number of tokens")
        plt.ylabel("Frequency")
        plt.grid(True)

        name = " Delexicalized" if self.delexicalize else ""
        plt.title(f"{name} Token distribution of contexts and targets")
        plt.savefig(
            self.project_root
            / self.out_root
            / f"model_size_calc_turns_{self.num_turns}_dialogs_{'_'.join(map(str, self.num_dialogs))}{name}.png"
        )

        self._plot_max_avg_graph(
            turns["turn_max_len"],
            [mean(vals) for vals in turns["turn_avg_len"].values()],
            x_axis,
            label_name="Turn",
            title="Turn Max, Avg Length",
        )
        self._plot_max_avg_graph(
            turns["turn_context_max_len"],
            [mean(vals) for vals in turns["turn_context_avg_len"].values()],
            x_axis,
            label_name="Context",
            title="Context Max, Avg Length",
        )
        self._plot_max_avg_graph(
            turns["turn_target_max_len"],
            [mean(vals) for vals in turns["turn_target_avg_len"].values()],
            x_axis,
            label_name="Target",
            title="Target Max, Avg Length",
        )

    def _plot_max_avg_graph(
        self,
        data_max,
        data_avg,
        x_axis,
        label_name: str = "Plot Label",
        title: str = "Plot Title",
        width=0.8,
    ):
        fig, ax = plt.subplots()
        ax.bar(
            x_axis,
            data_max.values(),
            label=f"{label_name} Max",
            width=width,
        )
        ax.bar(x_axis + width / 2, data_avg, width=width, label=f"{label_name} Avg")

        ax.autoscale(tight=True)
        ax.set_title(title)
        plt.xlabel("Number of turns")
        plt.ylabel("Number of tokens")
        plt.legend()

        name = " Delexicalized" if self.delexicalize else ""
        fig_dir = self.project_root / self.out_root / "max_avg"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            fig_dir
            / f"{label_name}_max_avg_tokens_{self.num_turns}_dialogs_{'_'.join(map(str, self.num_dialogs))}{name}.png"
        )
        plt.close()

    def schema_size(self):
        rows = self._get_simple_tod_rows()
        turn_schema_size = defaultdict(list)
        for row in tqdm(rows):
            a = 1

    def run(self):
        # self.plot_model_size()
        self.schema_size()


@hydra.main(config_path="../config/", config_name="data_model_exploration")
def hydra_start(cfg: DictConfig) -> None:
    msc = DataModelExploration(DataModelExplorationConfig(**cfg))
    msc.run()


if __name__ == "__main__":
    hydra_start()
