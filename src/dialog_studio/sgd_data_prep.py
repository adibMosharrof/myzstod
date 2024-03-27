import copy
import itertools
from pathlib import Path
import sys
import os
import numpy as np

from tqdm import tqdm

sys.path.insert(0, os.path.abspath("./src"))

from configs.dialog_studio_data_prep_config import DialogStudioDataPrepConfig
from tod.zs_tod_target import ZsTodTarget
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief
from tod.zs_tod_dst import ZsTodDst

from tod.turns.zs_tod_turn import ZsTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase
from tod.turns.turn_csv_row_factory import TurnCsvRowFactory
from tod.zs_tod_context import ZsTodContext

from torch.utils.data import Subset
import hydra
from omegaconf import DictConfig
from my_enums import Steps, ZsTodConstants
import utils

from pathos.multiprocessing import ProcessingPool as Pool
from sgd_dstc8_data_model.dstc_dataclasses import DstcDialog, DstcTurn
from datasets import load_dataset


class SgdDialogStudioDataPrep:
    def __init__(self, cfg: DialogStudioDataPrepConfig):
        self.cfg = cfg

    def run(self):
        ds = load_dataset("Salesforce/dialogstudio", "SGD")
        pass


@hydra.main(
    config_path="../../config/data_prep/", config_name="task_master_2_data_prep"
)
def hydra_start(cfg: DictConfig) -> None:
    tmwdp = SgdDialogStudioDataPrep(DialogStudioDataPrepConfig(**cfg))
    tmwdp.run()


if __name__ == "__main__":
    hydra_start()
