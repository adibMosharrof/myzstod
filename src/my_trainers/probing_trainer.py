import logging
import os
from pathlib import Path
import random
import sys
import uuid

from accelerate.accelerator import Accelerator
from dotmap import DotMap
import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.abspath("./src"))
sys.path.insert(0, os.path.abspath("./"))
from configs.base_trainer_config import BaseTrainerConfig
from generation.generation_handler_factory import GenerationHandlerFactory
from my_enums import Steps
from playground.t5_datamodule import T5DataModule
import utils
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas
from logger.results_logger import ResultsLogger
from metric_managers.metric_manager_factory import MetricManagerFactory
from my_trainers.base_trainer import BaseTrainer
from datamodules.tod_datamodulev2 import TodDataModuleV2
from tod.turns.zs_tod_turn import TodTurnApiCallCsvRow
from datamodules.tod_dataset import TodDataSet


class ProbingTrainer(BaseTrainer):
    def __init__(self, cfg: dict):
        super().__init__(cfg, dm_class=TodDataModuleV2)

    # def get_dm_dataset(self, dm):
    #     grouped_datasets = []
    #     for test_dataset in dm.datasets["test"]:
    #         self.separate_augmented_data(test_dataset, grouped_datasets)
    #     return {
    #         "train": dm.datasets["train"],
    #         "dev": dm.datasets["dev"],
    #         "test": grouped_datasets,
    #     }

    # def separate_augmented_data(self, orig_test_dataset, grouped_test):
    #     df = pd.DataFrame([item.__dict__ for item in orig_test_dataset])
    #     grouped_df = df.groupby("dataset_name")
    #     for group_name, group in grouped_df:
    #         csv_rows = [
    #             TodTurnApiCallCsvRow(**row) for row in group.to_dict(orient="records")
    #         ]
    #         test_dataset = TodDataSet(
    #             data=csv_rows,
    #             dataset_name=group_name,
    #             domain_setting=orig_test_dataset.domain_setting,
    #             step_name=orig_test_dataset.step_name,
    #             raw_data_root=orig_test_dataset.raw_data_root,
    #         )
    #         grouped_test.append(test_dataset)


@hydra.main(config_path="../../config/probing/", config_name="probing_trainer")
# @hydra.main(config_path="../../config/probing/", config_name="pseudo_trainer")
def hydra_start(cfg: DictConfig) -> None:
    # base_trainer_cfg = BaseTrainerConfig(**cfg)
    # ptrainer = ProbingTrainer(base_trainer_cfg)
    ptrainer = ProbingTrainer(cfg)
    ptrainer.run()


if __name__ == "__main__":
    hydra_start()
