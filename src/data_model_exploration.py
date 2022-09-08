from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

import dstc_utils
from dstc_dataclasses import DstcDomains, Steps
from my_datamodules import SimpleTodDataModule
from simple_tod_dataclasses import (
    SimpleTodConstants,
    SimpleTodDatasetItem,
    SimpleTodTurnCsvRow,
    SpecialTokens,
)
from utils import read_csv_dataclass


class DataModelExploration:
    def __init__(
        self,
        data_root: str = None,
        project_root: str = None,
        num_dialogs: list[int] = None,
        delexicalize: bool = False,
        model_name: str = "gpt2",
        out_root: str = "figures/model_size_calc",
        num_turns: int = 10,
        domain_settings: str = "SEEN",
        overwrite: list[bool] = None,
        data_split_percent: list[float] = None,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = self.project_root / out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.tokenizer = dstc_utils.get_tokenizer(model_name)
        special_tokens = SpecialTokens.list()
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.num_turns = num_turns
        self.domains = DstcDomains[domain_settings.upper()].value
        self.dm = SimpleTodDataModule(
            project_root=self.project_root,
            data_split_percent=data_split_percent,
            num_dialogs=[127, 20, 34],
            domains=DstcDomains.ALL.value,
            is_multi_task=True,
            overwrite=overwrite,
        )
        self.dm.setup()

    def _get_simple_tod_rows(self):
        steps = Steps.list()
        rows = []
        for step, num_dialog in tqdm(zip(steps, self.num_dialogs)):
            csv_file_path = dstc_utils.get_csv_data_path(
                step,
                num_dialog,
                delexicalized=False,
                processed_data_root=self.data_root,
                domains=self.domains,
            )
            rows.append(read_csv_dataclass(csv_file_path, SimpleTodTurnCsvRow))
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

    def plot_intents_distribution(self):
        rows = self._get_simple_tod_rows()

        a = 1

    def update_token_len(
        self,
        data: SimpleTodDatasetItem,
    ):
        c_tok = self.tokenizer(data.context, return_tensors="pt")
        t_tok = self.tokenizer(data.target, return_tensors="pt")
        c_tok_len = len(c_tok["input_ids"])
        t_tok_len = len(t_tok["input_ids"])
        self.context_hist[c_tok_len] += 1
        self.target_hist[t_tok_len] += 1

    def plot_multitask_token_len(self):
        self.context_hist = defaultdict(int)
        self.target_hist = defaultdict(int)
        for step in tqdm(Steps.list()):
            data = self.dm.multi_task_preprocessor(self.dm.datasets[step])
            # res = list(
            #     tqdm(
            #         Pool().imap(
            #             self.update_token_len,
            #             data,
            #         ),
            #         total=len(data),
            #     )
            # )
            for d in tqdm(data):
                c_tok = self.tokenizer(d.context, return_tensors="pt")
                t_tok = self.tokenizer(d.target, return_tensors="pt")
                c_tok_len = c_tok["input_ids"].shape[1]
                t_tok_len = t_tok["input_ids"].shape[1]
                self.context_hist[c_tok_len] += 1
                self.target_hist[t_tok_len] += 1
        print(
            f"Context len max:{max(self.context_hist.keys())}, avg {mean(self.context_hist.keys())}"
        )
        print(
            f"Target len max:{max(self.target_hist.keys())}, avg {mean(self.target_hist.keys())}"
        )
        a = 1

    def run(self):
        # self.plot_intents_distribution()
        # self.plot_model_size()
        self.plot_multitask_token_len()


@hydra.main(config_path="../config/", config_name="data_model_exploration")
def hydra_start(cfg: DictConfig) -> None:
    msc = DataModelExploration(
        project_root=cfg.project_root,
        data_root=cfg.data_root,
        delexicalize=cfg.delexicalize,
        num_dialogs=cfg.num_dialogs,
        model_name=cfg.model_name,
        out_root=cfg.out_root,
        num_turns=cfg.num_turns,
        domain_settings=cfg.domain_settings,
        data_split_percent=cfg.data_split_percent,
        overwrite=cfg.overwrite,
    )
    msc.run()


if __name__ == "__main__":
    hydra_start()
