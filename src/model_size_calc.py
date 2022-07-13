from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from simple_tod_dataclasses import SimpleTodTurnCsvRow, SpecialTokens
from utils import read_csv_dataclass
import numpy as np

import matplotlib.pyplot as plt


class ModelSizeCalc:
    def __init__(
        self,
        data_root: str = None,
        project_root: str = None,
        num_dialogs: int = 127,
        delexicalize: bool = True,
        model_name: str = "gpt2",
        out_root: str = "out/model_size_calc",
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = self.project_root / out_root
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        # special_tokens = torch.tensor(SpecialTokens.list(), device=torch.device("cuda"))
        special_tokens = SpecialTokens.list()
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

    def _get_simple_tod_rows(self):
        steps = ["train", "dev", "test"]
        rows = []
        for step in tqdm(steps):
            root = self.data_root / step
            if self.delexicalize:
                csv_path = (
                    root / f"simple_tod_dstc_{self.num_dialogs}_delexicalized.csv"
                )
            else:
                csv_path = root / f"simple_tod_dstc_{self.num_dialogs}.csv"
            rows.append(read_csv_dataclass(csv_path, SimpleTodTurnCsvRow))
        return np.concatenate(rows, axis=0)

    def plot_model_size(self):
        rows = self._get_simple_tod_rows()
        contexts = []
        targets = []
        for row in tqdm(rows):
            c = self.tokenizer(row.context, return_tensors="pt")
            t = self.tokenizer(row.target, return_tensors="pt")
            contexts.append(c.data["input_ids"].shape[1])
            targets.append(t.data["input_ids"].shape[1])
            # contexts.append(len())
            # targets.append(len(self.tokenizer(row.target, return_tensors="pt")))
        r = np.arange(len(rows))

        plt.scatter(r, contexts, color="b", label="Contexts", s=0.5)
        plt.scatter(r, targets, color="g", label="Targets", s=0.5)

        plt.xlabel("Index")
        plt.ylabel("Number of tokens")
        plt.legend()

        name = " Delexicalized" if self.delexicalize else ""
        plt.title(f"{name} Token distribution of contexts and targets")
        plt.savefig(
            self.project_root
            / self.out_root
            / f"model_size_calc_{self.num_dialogs}{name}.png"
        )


@hydra.main(config_path="../config/", config_name="model_size_calc")
def hydra_start(cfg: DictConfig) -> None:
    msc = ModelSizeCalc(
        project_root=cfg.project_root,
        data_root=cfg.data_root,
        delexicalize=cfg.delexicalize,
        num_dialogs=cfg.num_dialogs,
        model_name=cfg.model_name,
        out_root=cfg.out_root,
    )
    msc.plot_model_size()


if __name__ == "__main__":
    hydra_start()
