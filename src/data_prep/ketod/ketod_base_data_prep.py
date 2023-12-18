import sys

import numpy as np


sys.path.append("./src")
sys.path.append("./src/data_prep")

from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcDialog,
    DstcFrame,
    DstcSchema,
    DstcTurn,
    get_schemas,
)
from utilities.dialog_studio_dataclasses import DsDialog
from tod.turns.turn_csv_row_base import TurnCsvRowBase
from tod.turns.turn_csv_row_factory import TurnCsvRowFactory
import hydra
from omegaconf import DictConfig
from configs.dataprep_config import DataPrepConfig
import utils
from pathlib import Path
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from data_prep_strategy_resolver import DataPrepStrategyResolver
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas, DstcSchema, DstcDialog
from my_enums import Steps
from data_prep.data_prep_strategy import DataPrepStrategy
from datasets import load_dataset
import json
from dotmap import DotMap
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from tod.nlg.nlg_tod_turn import NlgTodTurn
import data_prep.data_prep_utils as data_prep_utils
from torch.utils.data import Subset


class KetodBaseDataPrep:
    def __init__(self, cfg: DataPrepConfig, data_prep_strategy: DataPrepStrategy):
        self.cfg = cfg
        self.data_prep_strategy = data_prep_strategy
        # if self.cfg.step_name == Steps.DEV.value:
        #     self.cfg.step_name = "validation"

    def prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, DstcSchema],
    ) -> NlgTodTarget:
        response = self._prepare_response(system_turn)
        return NlgTodTarget(response=response)

    def _prepare_dialog(self, dialog, schemas, turn_csv_row_handler):
        services = []
        for turn in dialog.log:
            for frame in turn.original_user_side_information.frames:
                services.append(frame.service)
        dialog.services = list(set(services))
        prepped_dialog = self.data_prep_strategy.prepare_dialog(
            dialog, schemas, turn_csv_row_handler
        )
        if prepped_dialog is None:
            return None
        return prepped_dialog

    def run(self):
        schemas = {}
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in Steps.list()]:
            schemas.update(d)
        csv_file_path = utils.get_csv_data_path(
            step=self.cfg.step_name, num_dialogs=self.cfg.num_dialogs, cfg=self.cfg
        )
        if csv_file_path.exists() and not self.cfg.overwrite:
            print(
                f"{self.cfg.step_name} csv file already exists and overwrite is false, so skipping"
            )
            return
        turn_csv_row_handler: TurnCsvRowBase = TurnCsvRowFactory.get_handler(self.cfg)
        out_data = []
        dataset = load_dataset("Salesforce/dialogstudio", "KETOD")
        if step_name == Steps.DEV.value:
            step_name = "validation"
        ds = dataset[self.cfg.step_name]
        if self.cfg.num_dialogs < 1:
            self.cfg.num_dialogs = len(ds)
        subset_data = Subset(ds, range(self.cfg.num_dialogs))
        for row in subset_data:
            dialog = DsDialog(row)
            prepped_dialog = self._prepare_dialog(dialog, schemas, turn_csv_row_handler)
            if prepped_dialog is None:
                continue
            out_data.append(prepped_dialog)
        headers = turn_csv_row_handler.get_csv_headers(self.cfg.should_add_schema)
        if len(out_data) == 0:
            print(f"No data for {self.cfg.step_name}")
            return
        csv_data = np.concatenate(out_data, axis=0)
        utils.write_csv(headers, csv_data, csv_file_path)


@hydra.main(
    config_path="../../../config/data_prep/", config_name="ketod_base_data_prep"
)
def hydra_start(cfg: DictConfig) -> None:
    dpconf = DataPrepConfig(**cfg)
    dp_strategy = DataPrepStrategyResolver.resolve(dpconf)
    stdp = KetodBaseDataPrep(dpconf, dp_strategy)
    stdp.run()


if __name__ == "__main__":
    hydra_start()
