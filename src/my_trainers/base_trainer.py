from dataclasses import asdict
import logging
import os
from pathlib import Path
import random
import sys
import uuid

from accelerate.accelerator import Accelerator
from dotmap import DotMap
import hydra
import numpy as np
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

from model_loaders.base_model_loader import BaseModelLoader


sys.path.insert(0, os.path.abspath("./src"))
sys.path.insert(0, os.path.abspath("./"))
from datamodules.data_collators.base_collator import BaseCollator
from datamodules.data_collators.collator_factory import CollatorFactory
from generation.generation_handler_factory import GenerationHandlerFactory
from my_enums import Steps
from playground.t5_datamodule import T5DataModule
import utils
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas
from logger.results_logger import ResultsLogger
from metric_managers.metric_manager_factory import MetricManagerFactory
from datamodules.data_filters.data_filter_registry import DATA_FILTER_MAP
from model_loaders.model_loader_factory import ModelLoaderFactory
from configs.base_trainer_config import BaseTrainerConfig
from tod.turns.zs_tod_turn import TodTurnCsvRowFactory
from configs.dm_config import DataModuleConfig
from prompts.nlg_prompt_manager import NlgPromptFactory


class BaseTrainer:
    def __init__(self, cfg: BaseTrainerConfig, dm_class=T5DataModule):
        self.dm_class = dm_class
        self.simplify_config(cfg)
        formatter = logging.Formatter(fmt="%(message)s")
        root_logger = logging.getLogger()  # no name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
        self.logger = root_logger

    

    def init_model(self, model_name: str, model_path: str = None):
        model_cls = utils.get_model_class(model_name)
        if model_path:
            model_path = self.cfg.project_root / model_path
        model = model_cls.from_pretrained(model_path or model_name)
        return model

    def run(self):
        accelerator = Accelerator()
        torch.manual_seed(420)
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_name or self.cfg.model_type.model_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )

        dms = self.get_data_modules(tokenizer)
        train_dataset, val_dataset, test_datasets = self.get_datasets_from_data_modules(
            dms
        )
        model_loader = ModelLoaderFactory.get_loader(self.cfg, tokenizer)
        prompt_cls = NlgPromptFactory.get_handler(
            self.cfg.prompt_type, self.cfg.model_type.context_type
        )
        collator = CollatorFactory.create_collator(
            model_name=self.cfg.model_type.model_name,
            tokenizer=tokenizer,
            prompt_cls=prompt_cls,
            max_token_len=self.cfg.max_token_len,
            test_prompt_max_len=self.cfg.test_prompt_max_len,
        )
        if self.cfg.should_train:
            model = self.init_model(model_name)
            model.resize_token_embeddings(len(tokenizer))
            deepspeed_path = str(self.cfg.project_root / "config/ds_zero_tod.json")
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.cfg.out_dir,
                num_train_epochs=self.cfg.epochs,
                logging_steps=30,
                save_total_limit=5,
                save_steps=self.cfg.data_size.save_steps,
                eval_steps=self.cfg.data_size.eval_steps,
                load_best_model_at_end=True,
                save_strategy=IntervalStrategy.STEPS,
                # evaluation_strategy=IntervalStrategy.STEPS,
                eval_strategy=IntervalStrategy.STEPS,
                per_device_train_batch_size=self.cfg.model_type.train_batch_size,
                per_device_eval_batch_size=self.cfg.model_type.eval_batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                dataloader_drop_last=True,
                dataloader_num_workers=1,
                gradient_accumulation_steps=self.cfg.model_type.gradient_accumulation_steps,
                eval_accumulation_steps=self.cfg.model_type.eval_accumulation_steps,
                learning_rate=self.cfg.model_type.learning_rate,
                gradient_checkpointing=True,
                ddp_find_unused_parameters=False,
                deepspeed=deepspeed_path,
                ddp_backend="nccl",
                save_safetensors=False,
                # gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=dms[0].tod_train_collate,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.cfg.early_stopping_patience
                    ),
                ],
            )
            if self.cfg.get("resume_checkpoint", None):
                trainer.train(str(self.cfg.project_root / self.cfg.resume_checkpoint))
            else:
                trainer.train()
            if accelerator.is_main_process:
                self.save_model(trainer)
            accelerator.wait_for_everyone()
            model_out_dir = self.cfg.out_dir
        else:
            model_out_dir = str(self.cfg.project_root / self.cfg.model_type.model_path)

        utils.log(self.logger, "starting inference")
        model = model_loader.load(model_out_dir)
        model.eval()
        collate_fn = collator.tod_test_collate
        generation_handler = GenerationHandlerFactory.get_handler(
            self.cfg, model, tokenizer
        )
        out_dir_path = Path(self.cfg.out_dir)
        for test_dataset, domain_names_list in zip(
            test_datasets, self.cfg.test_domain_settings
        ):
            domain_names = ",".join(domain_names_list)
            if not len(test_dataset):
                print(f"No data for {domain_names}")
                continue
            # print(f"testing {domain_names}")
            utils.log(self.logger, f"testing {domain_names}")

            test_dl = DataLoader(
                test_dataset,
                batch_size=self.cfg.model_type.test_batch_size,
                collate_fn=collate_fn,
                pin_memory=True,
                num_workers=8,
            )
            test_dl = accelerator.prepare(test_dl)
            metric_manager = MetricManagerFactory.get_metric_manager(
                self.cfg.model_type.context_type, tokenizer, self.logger
            )
            for batch in tqdm(test_dl):
                max_gen_len = self.cfg.max_token_len

                sample_outputs = generation_handler.get_generation(
                    batch,
                    max_gen_len - self.cfg.test_prompt_max_len,
                    max_gen_len,
                    self.cfg.test_prompt_max_len,
                    False,
                    accelerator,
                    metric_manager,
                )
            print("generation complete")
            # must call this first
            metric_manager.compute_row_wise_metrics()
            metric_manager.compute_metrics(domain_names)
            metric_manager.compute_is_retrieval_and_slot_fill_metrics()
            csv_path = Path(self.cfg.out_dir) / f"{domain_names}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            metric_manager.write_csv(csv_path)

        if accelerator.is_main_process:
            chatgpt_results = (
                self.cfg.project_root / "data_exploration/chatgpt/chat_gpt_all.csv"
            )
            rand_key = random.choice(list(self.cfg.dataset.keys()))
            rand_dataset = self.cfg.dataset[rand_key]
            all_results = []
            for domain_names_list in self.cfg.test_domain_settings:
                domain_names = ",".join(domain_names_list)
                csv_path = out_dir_path / f"{domain_names}.csv"
                data = pd.read_csv(csv_path)
                all_results.append(data)
            combined_results = pd.concat(all_results)
            combined_csv_path = out_dir_path / "combined_results.csv"
            combined_results.to_csv(combined_csv_path, index=False)
            rl = ResultsLogger(
                DotMap(
                    project_root=self.cfg.project_root,
                    results_path=combined_csv_path,
                    chatgpt_results_path=str(chatgpt_results),
                    out_dir=out_dir_path / "results_logger",
                    raw_data_root=rand_dataset.raw_data_root,
                )
            )
            rl.run()
        accelerator.wait_for_everyone()

    def train_model(
        self,
        accelerator: Accelerator,
        collator: BaseCollator,
        train_dataset: list[any],
        val_dataset: list[any],
        model_loader: BaseModelLoader,
    ):
        model = model_loader.load()
        deepspeed_path = str(self.cfg.project_root / "config/ds_zero_tod.json")
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.cfg.out_dir,
            num_train_epochs=self.cfg.epochs,
            logging_steps=10,
            save_total_limit=5,
            save_steps=self.cfg.data_size.save_steps,
            eval_steps=self.cfg.data_size.eval_steps,
            load_best_model_at_end=True,
            save_strategy=IntervalStrategy.STEPS,
            eval_strategy=IntervalStrategy.STEPS,
            per_device_train_batch_size=self.cfg.model_type.train_batch_size,
            per_device_eval_batch_size=self.cfg.model_type.eval_batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            dataloader_drop_last=True,
            dataloader_num_workers=1,
            gradient_accumulation_steps=self.cfg.model_type.gradient_accumulation_steps,
            eval_accumulation_steps=self.cfg.model_type.eval_accumulation_steps,
            learning_rate=self.cfg.model_type.learning_rate,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            deepspeed=deepspeed_path,
            ddp_backend="nccl",
            save_safetensors=False,
            # gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator.tod_train_collate,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                ),
            ],
        )
        trainer.train()
        if accelerator.is_main_process:
            self.save_model(trainer)
        accelerator.wait_for_everyone()
        return self.cfg.out_dir

    def save_model(self, trainer):
        trainer.model.save_pretrained(self.cfg.out_dir, safe_serialization=False)

    def init_dm_class(self, dm_cfg, tokenizer):
        data_filters = []
        if "data_filters" in dm_cfg:
            for filter_name in dm_cfg.data_filters:
                data_filters.append(DATA_FILTER_MAP[filter_name])
        tod_turn_row_cls = TodTurnCsvRowFactory.get_handler(self.cfg)
        return self.dm_class(
            DataModuleConfig(tokenizer=tokenizer, **dm_cfg),
            tod_turn_row_cls=tod_turn_row_cls,
            data_filters=data_filters,
        )

    def get_data_modules(self, tokenizer):
        all_dms = []
        for dataset_name, dataset_config in self.cfg.dataset.items():
            dm_cfg = DotMap(self.cfg)
            dm_cfg.update(**dataset_config)
            dm_cfg.update(**self.cfg.data_size)
            dm_cfg.update(**self.cfg.model_type)
            dm_cfg.raw_data_root = self.cfg.project_root / dataset_config.raw_data_root
            schemas = {}
            for d in [
                get_schemas(self.cfg.project_root / dataset_config.raw_data_root, step)
                for step in Steps.list()
            ]:
                schemas.update(d)
            dm = self.init_dm_class(dm_cfg, tokenizer)
            dm.setup()
            all_dms.append(dm)
        return all_dms

    def get_dm_dataset(self, dm):
        return dm.datasets

    def get_datasets_from_data_modules(self, dms):
        train, val, test = [], [], []
        for dm in dms:
            ds = self.get_dm_dataset(dm)
            # api_datasets = self.get_api_call_datasets(ds)
            train.append(ds["train"])
            val.append(ds["dev"])
            test.append(np.concatenate(ds["test"], axis=0))
        return (
            np.concatenate(train, axis=0),
            np.concatenate(val, axis=0),
            np.concatenate(test, axis=0),
        )

    def get_api_call_datasets(self, ds):
        out = {"train": [], "dev": [], "test": []}
        for key in out.keys():
            if key == "test":
                for item in ds[key]:
                    data = [i for i in item.data if i.turn_row_type == 1]
                    if len(data):
                        out[key].append(data)
            else:
                out[key] = [i for i in ds[key].data if i.turn_row_type == 1]
        return out

    def simplify_config(self, cfg):
        self.cfg = DotMap(cfg)
        self.cfg.project_root = Path(self.cfg.project_root)
        self.cfg.out_dir = str(os.getcwd() / Path(self.cfg.out_dir))
        # self.model_name = self.cfg.model_type.model_name
        # self.model_path = self.cfg.model_type.model_path
        # self.context_type = self.cfg.model_type.context_type
        # self.train_batch_size = self.cfg.model_type.train_batch_size
        # self.eval_batch_size = self.cfg.model_type.eval_batch_size
        # self.test_batch_size = self.cfg.model_type.test_batch_size
        # self.gradient_accumulation_steps = self.cfg.model_type.gradient_accumulation
        # self.eval_accumulation_steps = self.cfg.model_type.eval_accumulation
        # self.quantization = self.cfg.model_type.quantization
        # self.quantization_dtype = self.cfg.model_type.quantization_dtype
        # self.learning_rate = self.cfg.model_type.learning_rate
        # self.model_log_name = self.cfg.model_type.model_log_name
