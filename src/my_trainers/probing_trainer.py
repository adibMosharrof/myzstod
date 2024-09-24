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


class ProbingTrainer(BaseTrainer):
    def __init__(self, cfg: dict):
        super().__init__(cfg, dm_class=TodDataModuleV2)


@hydra.main(config_path="../../config/probing/", config_name="probing_trainer")
def hydra_start(cfg: DictConfig) -> None:
    # base_trainer_cfg = BaseTrainerConfig(**cfg)
    # ptrainer = ProbingTrainer(base_trainer_cfg)
    ptrainer = ProbingTrainer(cfg)
    ptrainer.run()


if __name__ == "__main__":
    hydra_start()
