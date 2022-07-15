from cProfile import label
from pathlib import Path
from typing import List
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from dstc_dataclasses import Steps

from simple_tod_dataclasses import (
    SimpleTodConstants,
    SimpleTodTurnCsvRow,
    SpecialTokens,
)
from utils import read_csv_dataclass
import numpy as np
import dstc_utils
import matplotlib.pyplot as plt
from statistics import mean


class ModelSizeCalc:
    def __init__(
        self,
        data_root: str = None,
        project_root: str = None,
        num_dialogs: int = 127,
        delexicalize: bool = True,
        model_name: str = "gpt2",
        out_root: str = "figures/model_size_calc",
        num_turns: int = 10,
        domains: List[str] = None,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = self.project_root / out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.tokenizer = dstc_utils.get_tokenizer(model_name)
        # special_tokens = torch.tensor(SpecialTokens.list(), device=torch.device("cuda"))
        special_tokens = SpecialTokens.list()
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.num_turns = num_turns
        self.domains = domains

    def _get_simple_tod_rows(self):
        steps = Steps.list()
        rows = []
        for step, num_dialog in tqdm(zip(steps, self.num_dialogs)):
            csv_file_path = (
                self.data_root
                / step
                / f"simple_tod_dstc_turns_{self.num_turns}_dialogs_{num_dialog}{SimpleTodConstants.DELEXICALIZED if self.delexicalize else ''}_{'_'.join(self.domains)}.csv"
            )
            rows.append(read_csv_dataclass(csv_file_path, SimpleTodTurnCsvRow))
        return np.concatenate(rows, axis=0)

    def plot_model_size(self):
        rows = self._get_simple_tod_rows()
        turn_max_len = {}
        turn_avg_len = {}
        token_freq = []
        for row in tqdm(rows):
            c = self.tokenizer(row.context, return_tensors="pt")
            t = self.tokenizer(row.target, return_tensors="pt")
            c_size = c.data["input_ids"].shape[1]
            t_size = t.data["input_ids"].shape[1]
            # contexts.append(c_size)
            # targets.append(t_size)
            text_len = c_size + t_size
            token_freq.append(text_len)
            turn_id = int(row.turn_id)
            cur_max = turn_max_len.get(turn_id, None)
            turn_avg = turn_avg_len.get(turn_id, None)
            if turn_avg is None:
                turn_avg = []
                turn_avg_len[turn_id] = turn_avg
            turn_avg.append(text_len)
            if cur_max is None:
                turn_max_len[turn_id] = text_len
            elif cur_max < text_len:
                turn_max_len[turn_id] = text_len

        fig1 = plt.hist(token_freq)

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

        fig2 = plt.figure()
        avg_turn_len = [mean(vals) for vals in turn_avg_len.values()]
        plt.bar(turn_max_len.keys(), turn_max_len.values(), label="Max")
        plt.bar(turn_max_len.keys(), avg_turn_len, label="Avg")

        plt.xlabel("Number of turns")
        plt.ylabel("Max number of tokens")
        plt.grid(True)
        plt.legend()
        name = " Delexicalized" if self.delexicalize else ""
        plt.title(f"{name} Max number of tokens per turn")
        plt.savefig(
            self.project_root
            / self.out_root
            / f"max_avg_tokens_per_turn_{self.num_turns}_dialogs_{'_'.join(map(str, self.num_dialogs))}{name}.png"
        )


@hydra.main(config_path="../config/", config_name="data_model_exploration")
def hydra_start(cfg: DictConfig) -> None:
    msc = ModelSizeCalc(
        project_root=cfg.project_root,
        data_root=cfg.data_root,
        delexicalize=cfg.delexicalize,
        num_dialogs=cfg.num_dialogs,
        model_name=cfg.model_name,
        out_root=cfg.out_root,
        num_turns=cfg.num_turns,
        domains=cfg.domains,
    )
    msc.plot_model_size()


if __name__ == "__main__":
    hydra_start()
