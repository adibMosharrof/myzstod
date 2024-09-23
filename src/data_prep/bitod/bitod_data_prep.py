from pathlib import Path
import sys

import numpy as np


sys.path.append("./src")
sys.path.append("./src/data_prep")
import hydra
from omegaconf import DictConfig
from configs.dataprep_config import DataPrepConfig
import utils
from data_prep.data_prep_strategy import DataPrepStrategy
from data_prep.data_prep_strategy_factory import DataPrepStrategyFactory
from datasets import load_dataset

from tod.turns.turn_csv_row_base import TurnCsvRowBase
from tod.turns.turn_csv_row_factory import TurnCsvRowFactory
import data_prep.data_prep_utils as data_prep_utils
from torch.utils.data import Subset

from data_prep.bitod.bitod_schema_prep import BitodSchemaPrep
from my_enums import ContextType, Steps

from utilities.dialog_studio_dataclasses import DsDialog

from sgd_dstc8_data_model.dstc_dataclasses import DstcSchema
from langdetect import detect


class BitodDataPrep:
    def __init__(self, cfg: DataPrepConfig, data_prep_strategy: DataPrepStrategy):
        self.cfg = cfg
        self.data_prep_strategy = data_prep_strategy

    def get_schema(self) -> dict[str, DstcSchema]:
        path = self.cfg.raw_data_root / self.cfg.step_name / "schema.json"
        json_data = utils.read_json(path)
        return {item["service_name"]: DstcSchema(**item) for item in json_data}

    def run(self):
        schemas = self.get_schema()
        csv_file_path = utils.get_csv_data_path(
            step=self.cfg.step_name, num_dialogs=self.cfg.num_dialogs, cfg=self.cfg
        )
        if csv_file_path.exists() and not self.cfg.overwrite:
            print(
                f"{self.cfg.step_name} csv file already exists and overwrite is false, so skipping"
            )
            return

        dataset = load_dataset("Salesforce/dialogstudio", "BiTOD")
        turn_csv_row_handler: TurnCsvRowBase = TurnCsvRowFactory.get_handler(self.cfg)
        ds = data_prep_utils.get_dialog_studio_step_data(self.cfg.step_name, dataset)
        en_row_ids = []
        for i, row in enumerate(ds):
            dialog = DsDialog(row)
            lang = detect(dialog.log[0].user_utterance)
            if lang == "en":
                en_row_ids.append(i)
        en_data = Subset(ds, en_row_ids)

        if self.cfg.num_dialogs < 1:
            self.cfg.num_dialogs = len(en_data)
        subset_data = Subset(en_data, range(self.cfg.num_dialogs))
        out_data = []
        for row in subset_data:
            dialog = DsDialog(row)
            prepped_dialog = self.data_prep_strategy.prepare_dialog(
                dialog, schemas=schemas, turn_csv_row_handler=turn_csv_row_handler
            )
            if prepped_dialog is None:
                continue
            out_data.append(prepped_dialog)
        headers = turn_csv_row_handler.get_csv_headers(self.cfg.should_add_schema)
        if len(out_data) == 0:
            print(f"No data for {self.cfg.step_name}")
            return
        csv_data = np.concatenate(out_data, axis=0)
        utils.write_csv(headers, csv_data, csv_file_path)


@hydra.main(config_path="../../../config/data_prep/", config_name="bitod_data_prep")
def hydra_start(cfg: DictConfig) -> None:
    path = Path(cfg.project_root) / cfg.raw_data_root / cfg.step_name / "schema.json"
    if not path.exists() or cfg.overwrite:
        btsp = BitodSchemaPrep(cfg)
        btsp.run()
    dpconf = DataPrepConfig(**cfg)
    dp_strategy = DataPrepStrategyFactory.get_strategy(dpconf, ContextType.BITOD)
    stdp = BitodDataPrep(dpconf, dp_strategy)
    stdp.run()


if __name__ == "__main__":
    hydra_start()
