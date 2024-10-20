import pandas as pd
from schema.schema_loader import SchemaLoader
import sys


sys.path.append("./src")
sys.path.append("./src/data_prep")
import itertools
from typing import Dict

import hydra
from omegaconf import DictConfig
from configs.dataprep_config import DataPrepConfig
from pathlib import Path
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from data_prep.data_prep_strategy_factory import DataPrepStrategyFactory
from my_enums import Steps
from utils import get_csv_data_path, get_dialog_file_paths
from tod.turns.turn_csv_row_base import TurnCsvRowBase
from tod.turns.turn_csv_row_factory import TurnCsvRowFactory
import utils
import numpy as np
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas, DstcSchema, DstcDialog
from data_prep.data_prep_strategy import DataPrepStrategy


class DstcBaseDataPrep:
    def __init__(
        self,
        cfg: DataPrepConfig,
        data_prep_strategy: DataPrepStrategy,
        schema_loader: SchemaLoader = None,
        schemas: Dict[str, DstcSchema] = None,
    ):
        self.cfg = cfg
        self.data_prep_strategy = data_prep_strategy
        self.schemas = schemas or schema_loader.get_schemas(self.cfg.raw_data_root)

    def _prepare_dialog_file(
        self,
        path: Path,
        schemas: Dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> np.ndarray:
        data = []
        dialog_json_data = utils.read_json(path)
        for d in dialog_json_data:
            dialog = DstcDialog.from_dict(d)
            prepped_dialog = self.data_prep_strategy.prepare_dialog(
                dialog, schemas, turn_csv_row_handler
            )
            if prepped_dialog is None:
                continue
            data.append(prepped_dialog)
        if not len(data):
            return pd.DataFrame()
            # return np.array(data)
        # return np.concatenate(data, axis=0)
        turn_dfs = [pd.DataFrame(d) for d in data]
        conc_data = pd.concat(turn_dfs, axis=0, ignore_index=True)
        return conc_data

    def run(self):
        turn_csv_row_handler: TurnCsvRowBase = TurnCsvRowFactory.get_handler(self.cfg)
        step_dir = Path(self.cfg.processed_data_root / self.cfg.step_name)
        step_dir.mkdir(parents=True, exist_ok=True)
        dialog_paths = get_dialog_file_paths(self.cfg.raw_data_root, self.cfg.step_name)
        out_data = []
        if self.cfg.num_dialogs == "None":
            self.cfg.num_dialogs = len(dialog_paths)
        csv_file_path = get_csv_data_path(
            step=self.cfg.step_name,
            num_dialogs=self.cfg.num_dialogs,
            cfg=self.cfg,
        )
        if csv_file_path.exists() and not self.cfg.overwrite:
            print(
                f"{self.cfg.step_name} csv file already exists and overwrite is false, so skipping"
            )
            return

        if self.cfg.data_prep_multi_process:
            res = list(
                tqdm(
                    Pool().imap(
                        self._prepare_dialog_file,
                        dialog_paths[: self.cfg.num_dialogs],
                        itertools.repeat(self.schemas),
                        itertools.repeat(turn_csv_row_handler),
                    ),
                    total=self.cfg.num_dialogs,
                )
            )
        else:
            res = []
            for d in tqdm(dialog_paths[: self.cfg.num_dialogs]):
                output = self._prepare_dialog_file(
                    d, self.schemas, turn_csv_row_handler
                )
                if res is not None:
                    res.append(output)

        # out_data = [d for d in res if len(d)]

        headers = turn_csv_row_handler.get_csv_headers(self.cfg.should_add_schema)
        # csv_data = np.concatenate(out_data, axis=0)
        # utils.write_csv(headers, csv_data, csv_file_path)
        csv_data = pd.concat(res, axis=0)
        if csv_data.empty:
            domains = ",".join(self.cfg.domain_setting)
            print(f"No data for {self.cfg.step_name}: {domains}")
            return
        csv_data.to_csv(csv_file_path, index=False, header=headers)


@hydra.main(config_path="../../config/data_prep/", config_name="dstc_base_data_prep")
def hydra_start(cfg: DictConfig) -> None:
    dpconf = DataPrepConfig(**cfg)
    dp_strategy = DataPrepStrategyFactory.get_strategy(dpconf, dpconf.context_type)
    stdp = DstcBaseDataPrep(dpconf, dp_strategy)
    stdp.run()


if __name__ == "__main__":
    hydra_start()
