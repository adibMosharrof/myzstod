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
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
import torch


class CrossTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, dm_class=CrossDataModule)
        encoder_model = AutoModel.from_pretrained(self.cfg.encoder_model_name)
        accelerator = Accelerator()
        self.encoder_model = accelerator.prepare_model(encoder_model)
        # print(f"encoder model dtype: {self.encoder_model.dtype}")

    def init_model(self, model_name: str, model_path: str = None):
        if model_path:
            model = CustomGPT2Model.from_pretrained(model_path)
            model.encoder_model = self.encoder_model
            return model

        config = AutoConfig.from_pretrained(model_name)
        config.add_cross_attention = True
        # model = AutoModelForCausalLM.from_config(config)
        model = CustomGPT2Model(config, self.encoder_model)
        # model = accelerator.prepare_model(model)
        # print(f"model dtype: {model.dtype}")
        return model

    def init_dm_class(self, dm_cfg, tokenizer, schemas):
        return self.dm_class(dm_cfg, tokenizer, schemas, self.encoder_model)


class CustomGPT2Model(GPT2LMHeadModel):
    def __init__(self, config, encoder_model=None):
        # Call the parent class (GPT2LMHeadModel) init method
        super().__init__(config)

        # Initialize custom parameters
        self.encoder_model = encoder_model

    # Override the forward method to use custom logic
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        schema_tokens=None,
    ):

        # Custom behavior using custom parameters

        encoder_hidden_states = self.encoder_model(
            input_ids=schema_tokens
        ).last_hidden_state
        # Call the original forward method of GPT2LMHeadModel
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            encoder_hidden_states=encoder_hidden_states,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        schema_tokens=None,
        **kwargs
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, inputs_embeds, **kwargs
        )
        model_inputs["schema_tokens"] = schema_tokens
        return model_inputs


@hydra.main(config_path="../../config/probing/", config_name="cross_trainer")
def hydra_start(cfg: DictConfig) -> None:
    ctrainer = CrossTrainer(cfg)
    ctrainer.run()


if __name__ == "__main__":
    hydra_start()
