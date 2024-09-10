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
from datamodules.cross_datamodule import CrossDataModule
from generation.generation_handler_factory import GenerationHandlerFactory
from my_enums import Steps
from playground.t5_datamodule import T5DataModule
import utils
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas
from logger.results_logger import ResultsLogger
from metric_managers.metric_manager_factory import MetricManagerFactory
from base_trainer import BaseTrainer
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


class CrossTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, dm_class=CrossDataModule)
        self.encoder_model = AutoModel.from_pretrained(self.cfg.encoder_model_name)

    def init_model(self, model_name: str, model_path: str = None):
        if model_path:
            return AutoModelForCausalLM.from_pretrained(model_path).cuda()
        config = AutoConfig.from_pretrained(model_name)
        config.add_cross_attention = True
        model = AutoModelForCausalLM.from_config(config)
        return model.cuda()

    def init_dm_class(self, dm_cfg, tokenizer, schemas):
        return self.dm_class(dm_cfg, tokenizer, schemas, self.encoder_model)


@hydra.main(config_path="../../config/probing/", config_name="cross_trainer")
def hydra_start(cfg: DictConfig) -> None:
    ctrainer = CrossTrainer(cfg)
    ctrainer.run()


if __name__ == "__main__":
    hydra_start()
