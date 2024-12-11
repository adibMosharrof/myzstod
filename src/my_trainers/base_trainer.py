from dataclasses import asdict

from datamodules.tod_dataset import TodDataSet
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


sys.path.insert(0, os.path.abspath("./src"))
sys.path.insert(0, os.path.abspath("./"))
from model_loaders.base_model_loader import BaseModelLoader
from schema.schema_loader import SchemaLoader
from utilities.context_manager import ContextManager
from utilities.tokenizer_utilities import TokenizerUtilities
from datamodules.data_collators.base_collator import BaseCollator
from datamodules.data_collators.collator_factory import CollatorFactory
from generation.generation_handler_factory import GenerationHandlerFactory
from my_enums import SpecialTokens, Steps
from playground.t5_datamodule import T5DataModule
import utils
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas
from logger.results_logger import ResultsLogger
from metric_managers.metric_manager_factory import MetricManagerFactory
from model_loaders.model_loader_factory import ModelLoaderFactory
from configs.base_trainer_config import BaseTrainerConfig
from tod.turns.zs_tod_turn import TodTurnCsvRowFactory
from configs.dm_config import DataModuleConfig
from prompts.nlg_prompt_manager import NlgPromptFactory
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
)
from validators.should_add_schema_validator import ShouldAddSchemaValidator
from validators.should_train_validator import ShouldTrainValidator
from validators.target_length_validator import TargetLengthValidator
from datamodules.data_augmentation.data_augmentation_factory import (
    DataAugmentationFactory,
)
from datamodules.data_filters.data_filter_factory import (
    DataFilterFactory,
)


class BaseTrainer:
    def __init__(self, cfg: BaseTrainerConfig, dm_class=T5DataModule):
        self.dm_class = dm_class
        self.cfg = DotMap(cfg)
        self.cfg.project_root = Path(self.cfg.project_root)
        self.cfg.out_dir = str(os.getcwd() / Path(self.cfg.out_dir))
        formatter = logging.Formatter(fmt="%(message)s")
        root_logger = logging.getLogger()  # no name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
        self.logger = root_logger
        validators = [
            ShouldAddSchemaValidator(),
            ShouldTrainValidator(),
            TargetLengthValidator(),
        ]
        for validator in validators:
            validator.validate(self.cfg)

    def run(self):
        accelerator = Accelerator()
        torch.manual_seed(420)
        tokenizer = TokenizerUtilities.get_tokenizer(
            model_name=self.cfg.model_type.model_name,
            context_type=self.cfg.model_type.context_type,
        )

        model_loader = ModelLoaderFactory.get_loader(self.cfg, tokenizer)
        prompt_cls = NlgPromptFactory.get_handler(
            self.cfg.prompt_type, self.cfg.model_type.context_type
        )
        collator = CollatorFactory.create_collator(
            model_name=self.cfg.model_type.model_name,
            context_type=self.cfg.model_type.context_type,
            tokenizer=tokenizer,
            prompt_cls=prompt_cls,
            max_token_len=self.cfg.max_token_len,
            test_prompt_max_len=self.cfg.test_prompt_max_len,
            schema_max_len=self.cfg.get("schema_max_len", 350),
        )

        dms = self.get_data_modules(tokenizer, collator)
        train_dataset, val_dataset, test_datasets = self.get_datasets_from_data_modules(
            dms
        )
        # self.test_dm(train_dataset, collator.tod_train_collate)

        if self.cfg.should_train:
            model_out_dir = self.train_model(
                accelerator, collator, train_dataset, val_dataset, model_loader
            )
        else:
            model_out_dir = str(self.cfg.project_root / self.cfg.model_type.model_path)
        if not self.cfg.should_test:
            utils.log(self.logger, "should test is set to false, exiting")
            return
        utils.log(self.logger, "starting inference")
        model = model_loader.load_for_inference(model_out_dir)

        collate_fn = collator.tod_test_collate
        generation_handler = GenerationHandlerFactory.get_handler(
            self.cfg, model, tokenizer
        )
        out_dir_path = Path(self.cfg.out_dir)
        for test_dataset in test_datasets:
            domain_names = test_dataset.get_domain_names()
            if not len(test_dataset):
                print(f"No data for {domain_names}")
                continue
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
            csv_path = self.get_pred_csv_path(test_dataset)
            metric_manager.write_csv(csv_path)

        if accelerator.is_main_process:
            for test_dataset in test_datasets:
                self.log_results(test_dataset, out_dir_path)
        accelerator.wait_for_everyone()

    def get_pred_csv_path(self, test_dataset: TodDataSet):
        csv_path = (
            Path(self.cfg.out_dir)
            / "predictions"
            / f"{test_dataset.dataset_name}_{test_dataset.get_domain_names()}.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        return csv_path

    def log_results(self, dataset: TodDataSet, out_dir_path: Path):
        chatgpt_results = (
            self.cfg.project_root / "data_exploration/chatgpt/chat_gpt_all.csv"
        )
        csv_path = self.get_pred_csv_path(dataset)
        rl = ResultsLogger(
            DotMap(
                project_root=self.cfg.project_root,
                results_path=csv_path,
                chatgpt_results_path=str(chatgpt_results),
                out_dir=out_dir_path / "results_logger" / dataset.dataset_name,
                raw_data_root=dataset.raw_data_root,
                dataset_name=dataset.dataset_name,
            )
        )
        rl.run()

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
        bf16 = False
        fp16 = False
        if torch.cuda.is_bf16_supported():
            bf16 = True
        else:
            fp16 = True
        run_name = Path(self.cfg.out_dir).parent.name
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
            dataloader_num_workers=4,
            gradient_accumulation_steps=self.cfg.model_type.gradient_accumulation_steps,
            eval_accumulation_steps=self.cfg.model_type.eval_accumulation_steps,
            learning_rate=self.cfg.model_type.learning_rate,
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            deepspeed=deepspeed_path,
            ddp_backend="nccl",
            save_safetensors=False,
            report_to="wandb",
            run_name=run_name,
            # gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16_full_eval=bf16,
            bf16=bf16,
            fp16=fp16,
            fp16_full_eval=fp16,
            optim="paged_adamw_8bit",
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
        if self.cfg.resume_checkpoint:
            trainer.train(str(self.cfg.project_root / self.cfg.resume_checkpoint))
        else:
            trainer.train()
        if accelerator.is_main_process:
            self.save_model(trainer)
        accelerator.wait_for_everyone()
        return self.cfg.out_dir

    def save_model(self, trainer):
        trainer.model.save_pretrained(self.cfg.out_dir, safe_serialization=False)

    def init_dm_class(self, dm_cfg, tokenizer, collator):
        data_augmentations = []
        data_filters = (
            [
                DataFilterFactory.get_data_filter(
                    filter_name, cfg=dm_cfg, collator=collator
                )
                for filter_name in dm_cfg.data_filters
            ]
            if "data_filters" in dm_cfg
            else []
        )

        schema_loader = SchemaLoader(DstcSchema)
        schemas = schema_loader.get_schemas(dm_cfg.raw_data_root)
        data_augmentations = DataAugmentationFactory.create_data_augmentations(
            dm_cfg, schemas
        )
        dm_cfg.data_augmentations = data_augmentations
        tod_turn_row_cls = TodTurnCsvRowFactory.get_handler(self.cfg)
        return self.dm_class(
            DataModuleConfig(tokenizer=tokenizer, **dm_cfg),
            tod_turn_row_cls=tod_turn_row_cls,
            data_filters=data_filters,
            data_augmentations=data_augmentations,
            schemas=schemas,
        )

    def test_dm(self, dataset, collate_fn):
        for item in dataset:
            collate_fn([item])
        return

    def get_data_modules(self, tokenizer, collator):
        all_dms = []

        for dataset_name, dataset_config in self.cfg.dataset.items():
            dm_cfg = DotMap(self.cfg)
            dm_cfg.update(**dataset_config)
            dm_cfg.update(**self.cfg.data_size)
            dm_cfg.update(**self.cfg.model_type)
            dm_cfg.raw_data_root = self.cfg.project_root / dataset_config.raw_data_root
            dm = self.init_dm_class(dm_cfg, tokenizer, collator)
            dm.setup()
            all_dms.append(dm)
        return all_dms

    def get_dm_dataset(self, dm):
        return dm.datasets

    def get_datasets_from_data_modules(
        self, dms
    ) -> tuple[TodDataSet, TodDataSet, list[TodDataSet]]:
        train, val, test = [], [], []
        for dm in dms:
            ds = self.get_dm_dataset(dm)
            train.extend(ds["train"])
            val.extend(ds["dev"])
            test.extend(ds["test"])
            # train, val, test = self.get_dm_dataset(dm)
            # train.extend(train)
            # val.extend(val)
            # test.extend(test)
        return train, val, test

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
