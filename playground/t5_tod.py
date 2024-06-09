import os
import sys
import uuid

from omegaconf import DictConfig


sys.path.insert(0, os.path.abspath("./src"))
from generation.generation_handler_factory import GenerationHandlerFactory
from simple_tod_dataclasses import TodTestDataBatch
from metric_managers.metric_manager_factory import MetricManagerFactory
from metric_managers.ketod_metric_manager import KeTodMetricManager
from prompts.prompt_constants import NlgPromptType
from logger.service_call_inference_logger import ServiceCallInferenceLogger
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from my_enums import Steps, ContextType
from prompts.nlg_prompt_manager import NlgPromptFactory
from tod_datamodules import TodDataModule
from t5_datamodule import T5DataModule
import hydra
from logger.inference_logger import InferenceLogger
from metric_managers.nlg_metric_manager import NlgMetricManager
from tod.turns.zs_tod_turn import TodTurnCsvRow, TodTurnCsvRowFactory
from base_datamodule import SimpleTodDataSet
from pathlib import Path
from dotmap import DotMap
from torch.utils.data import DataLoader
import pandas as pd
from accelerate import Accelerator
import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    IntervalStrategy,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from datetime import datetime
import utils
from tqdm import tqdm
import numpy as np
import evaluate
import logging
from peft import (
    prepare_model_for_kbit_training,
    PeftModelForSeq2SeqLM,
    get_peft_config,
    PeftConfig,
    PeftModel,
    TaskType,
    LoraConfig,
    get_peft_model,
)
from accelerate import Accelerator
import deepspeed
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas

from metric_managers.bitod_metric_manager import BitodMetricManager
import wandb

from logger.results_logger import ResultsLogger

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


class T5Tod:
    def __init__(self, cfg):
        self.cfg = DotMap(cfg)
        self.cfg.project_root = Path(cfg.project_root)
        self.cfg.raw_data_root = self.cfg.project_root / self.cfg.dataset.raw_data_root
        self.cfg.context_type = self.cfg.dataset.context_type
        self.cfg.data_prep_out_root = self.cfg.dataset.data_prep_out_root
        self.cfg.num_dialogs = self.cfg.data_size.num_dialogs
        self.cfg.data_split_percent = self.cfg.data_size.data_split_percent

        log_file = self.cfg.project_root / self.cfg.out_dir / "t5_tod.log"
        # logging.basicConfig(
        #     filename=str(log_file), level=logging.INFO, encoding="utf-8", format='%(message)s'
        # )
        formatter = logging.Formatter(fmt="%(message)s")
        root_logger = logging.getLogger()  # no name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
        self.logger = root_logger
        # self.logger = logging
        self.cfg.out_dir = Path("results")

    def __old_init__(self, cfg, **kwargs):
        self.cfg = DotMap(dict(cfg))
        self.cfg.project_root = Path(self.cfg.project_root)
        base_out_dir = self.cfg.project_root / "playground" / "t5_tod_out"
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        self.cfg.out_dir = base_out_dir / date_str / time_str
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.cfg.out_dir)
        self.cfg.project_root = Path(self.cfg.project_root)
        self.cfg.out_dir = Path(os.getcwd())
        log_file = self.cfg.out_dir / "t5_tod.log"
        logging.basicConfig(filename=log_file, level=logging.INFO, encoding="utf-8")
        self.logger = logging
        self.cfg.raw_data_root = self.cfg.project_root / self.cfg.raw_data_root
        self.logger.info(self.cfg)
        print(self.cfg)

    def get_metric_manager(self, context_type: str, tokenizer):
        if context_type in [
            ContextType.NLG_API_CALL.value,
            ContextType.GPT_API_CALL.value,
        ]:
            return NlgApiCallMetricManager(self.logger, tokenizer)
        if context_type == ContextType.KETOD_API_CALL.value:
            return KeTodMetricManager(self.logger, tokenizer)
        if context_type == ContextType.BITOD.value:
            return BitodMetricManager(self.logger, tokenizer)
        return NlgMetricManager(self.logger, tokenizer)

    def pad_gen_to_max_len(self, gen, max_len: int, tokenizer):
        pad_amount = max_len - gen.shape[1]
        pad = torch.full(
            [gen.shape[0], pad_amount],
            fill_value=tokenizer.pad_token_id,
            dtype=torch.int,
            device=gen.device,
        )
        out = torch.hstack([gen, pad])
        return out

    def get_lora_task_type(self, model_class):
        if model_class == T5ForConditionalGeneration:
            return TaskType.SEQ_2_SEQ_LM
        return TaskType.CAUSAL_LM

    def get_model(self, model_name: str, model_class: any, tokenizer: AutoTokenizer):
        model_path = model_name
        if self.cfg.model_type.model_path:
            model_path = self.cfg.project_root / self.cfg.model_type.model_path
        if self.cfg.model_type.quantization_dtype == 16:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            model = model_class.from_pretrained(
                # model_path, load_in_8bit=True, torch_dtype=torch.bfloat16
                model_path,
                load_in_8bit=False,
                torch_dtype=dtype,
                # model_path, load_in_8bit=False
            )
            model.resize_token_embeddings(len(tokenizer))
            model = prepare_model_for_kbit_training(model)
            lora_config = utils.get_lora_config(model_name)
            model = get_peft_model(model, lora_config)
            return model
        if self.cfg.model_type.quantization_dtype == 8:
            config = BitsAndBytesConfig(load_in_8bit=True)
            device_map = {"": Accelerator().process_index}
            model = model_class.from_pretrained(
                model_path, quantization_config=config, device_map=device_map
            )
            model.enable_input_require_grads()
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            return model
        if self.cfg.model_type.quantization_dtype == 4:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_map = {"": Accelerator().process_index}
            model = model_class.from_pretrained(
                model_path, quantization_config=config, device_map=device_map
            )
            # model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=16,
                lora_alpha=8,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            return model

    def get_model_class(self, model_name: str):
        if "t5" in model_name:
            return T5ForConditionalGeneration
        lmhead_models = [
            "alpaca",
            "llama",
            "santacoder",
            "vicuna",
            "koala",
            "opt",
            "gpt",
        ]
        if any([m in model_name.lower() for m in lmhead_models]):
            return AutoModelForCausalLM
        raise ValueError(f"model_name {model_name} not supported")

    def run(self):
        accelerator = Accelerator()
        torch.manual_seed(420)

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_name or self.cfg.model_type.model_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<SYSTEM>", "<USER>", "{", "}"]}
        )
        tokenizer.model_max_length = 1024
        model = None
        deepspeed_path = None
        # tokenizer.add_tokens(["<|user|>", "<|system|>"])
        model_cls = self.get_model_class(self.cfg.model_type.model_name)
        if self.cfg.model_type.model_path:
            model_out_dir = str(self.cfg.project_root / self.cfg.model_type.model_path)
        elif self.cfg.model_type.quantization:
            model = self.get_model(self.cfg.model_type.model_name, model_cls, tokenizer)
            # deepspeed_path = self.cfg.project_root / "config/ds_zero1.json"
        else:
            # model = T5ForConditionalGeneration.from_pretrained(
            model = model_cls.from_pretrained(self.cfg.model_type.model_name).cuda()
            model.resize_token_embeddings(len(tokenizer))
        steps = Steps.list()
        schemas = {}
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in steps]:
            schemas.update(d)
        self.dm = T5DataModule(self.cfg, tokenizer, schemas)
        if self.cfg.separate_dev_test:
            train_dataset, val_dataset, test_dataset = self.dm.load_data_from_files()
        else:
            train_dataset, val_dataset, test_datasets = self.dm.load_data()

        if not self.cfg.model_type.model_path and self.cfg.should_train:
            bf16 = False
            fp16 = False
            if torch.cuda.is_bf16_supported():
                bf16 = True
            else:
                fp16 = True
            training_args = Seq2SeqTrainingArguments(
                output_dir=str(self.cfg.out_dir),
                num_train_epochs=self.cfg.epochs,
                logging_steps=10,
                save_total_limit=5,
                save_steps=self.cfg.data_size.save_steps,
                eval_steps=self.cfg.data_size.eval_steps,
                load_best_model_at_end=True,
                save_strategy=IntervalStrategy.STEPS,
                evaluation_strategy=IntervalStrategy.STEPS,
                per_device_train_batch_size=self.cfg.model_type.train_batch_size,
                per_device_eval_batch_size=self.cfg.model_type.eval_batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                dataloader_drop_last=True,
                dataloader_num_workers=1,
                gradient_accumulation_steps=self.cfg.model_type.gradient_accumulation_steps,
                eval_accumulation_steps=self.cfg.model_type.eval_accumulation_steps,
                learning_rate=1e-3,
                bf16_full_eval=bf16,
                bf16=bf16,
                fp16=fp16,
                fp16_full_eval=fp16,
                gradient_checkpointing=True,
                ddp_find_unused_parameters=False,
                deepspeed=deepspeed_path,
                # gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=self.dm.tod_train_collate,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.cfg.early_stopping_patience
                    ),
                ],
            )
            # model.gradient_checkpointing_disable()
            # with accelerator.no_sync(model):
            # trainer.train()

            if self.cfg.resume_checkpoint:
                trainer.train(str(self.cfg.project_root / self.cfg.resume_checkpoint))
            else:
                trainer.train()
            if accelerator.is_main_process:
                trainer.model.save_pretrained(self.cfg.out_dir)
            accelerator.wait_for_everyone()
            # trainer.save_model()
            # model.save_pretrained(self.cfg.out_dir)
            model_out_dir = self.cfg.out_dir

        if not self.cfg.model_type.model_path:
            _ = model.eval()
        print("starting inference")

        if (
            self.cfg.model_type.quantization
            and self.cfg.model_type.quantization_dtype == 16
        ):
            config = PeftConfig.from_pretrained(model_out_dir)
            device_map = {"": accelerator.device}
            model = model_cls.from_pretrained(
                config.base_model_name_or_path,
                load_in_8bit=False,
                # load_in_8bit=True,
                device_map=device_map,
            )
            model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(
                model, model_out_dir, device_map=device_map
            )
        elif (
            self.cfg.model_type.quantization
            and self.cfg.model_type.quantization_dtype == 4
        ):
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            device_map = {"": accelerator.device}
            model = model_cls.from_pretrained(
                self.cfg.model_type.model_name,
                quantization_config=config,
                device_map=device_map,
            )
            model = PeftModel.from_pretrained(model, model_out_dir)
        elif (
            self.cfg.model_type.quantization
            and self.cfg.model_type.quantization_dtype == 8
        ):  
            config=BitsAndBytesConfig(load_in_8bit=True)

            device_map = {"": accelerator.device}
            model = model_cls.from_pretrained(
                self.cfg.model_type.model_name,
                quantization_config=config,
                device_map=device_map,
            )
            model = PeftModel.from_pretrained(model, model_out_dir)
        model.eval()

        collate_fn = (
            self.dm.tod_test_collate
            if self.cfg.context_type
            in [
                ContextType.NLG_API_CALL.value,
                ContextType.GPT_API_CALL.value,
                ContextType.KETOD_API_CALL.value,
                ContextType.BITOD.value,
            ]
            else self.dm.tod_train_collate
        )

        generation_handler = GenerationHandlerFactory.get_handler(
            self.cfg, model, tokenizer
        )
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
            # metric_manager = self.get_metric_manager(self.cfg.context_type, tokenizer)
            metric_manager = MetricManagerFactory.get_metric_manager(
                self.cfg.context_type, tokenizer, self.logger
            )
            for batch in tqdm(test_dl):
                # max_gen_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
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
            # must call this first
            metric_manager.compute_row_wise_metrics()
            metric_manager.compute_metrics(domain_names)
            metric_manager.compute_is_retrieval_and_slot_fill_metrics()
            csv_path = self.cfg.out_dir / f"{domain_names}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            metric_manager.write_csv(csv_path)

        if accelerator.is_main_process:
            rl = ResultsLogger(
                DotMap(
                    project_root=self.cfg.project_root,
                    results_path=os.getcwd() / self.cfg.out_dir / "all.csv",
                    chatgpt_results_path="data_exploration/chatgpt/chat_gpt_all.csv",
                    out_dir=self.cfg.out_dir,
                    raw_data_root=self.cfg.raw_data_root
                )
            )
            rl.run()
        # for table in tables:
        #     self.logger.info(table)
        #     self.logger.info("\n\n")
        accelerator.wait_for_everyone()


# if __name__ == "__main__":
def old_main():
    tt = T5Tod(
        DotMap(
            # csv_file="nlg_data.csv",
            csv_file="v0_context_type_nlg_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_False_turns_26_service_results_True_dialogs_1_domain_setting_all_train_domains_1.0.csv",
            separate_dev_test=True,
            # project_root=Path("/projects/bbyl/amosharrof/ZSToD"),
            project_root=Path("/mounts/u-amo-d1/adibm-data/projects/ZSToD/"),
            # tokenizer_name="adibm/sgd-flan-t5-nlg-tokenizer",
            model_name="google/flan-t5-large",
            # model_path="playground/t5_tod_out/2023-10-27/00-42-59",
            # model_path="outputs/2023-10-25/11-49-15/results/pretrain",
            model_path="",
            max_token_len=1024,
            test_prompt_max_len=750,
            train_batch_size=10,
            eval_batch_size=20,
            test_batch_size=20,
            epochs=2,
            gradient_accumulation_steps=64,
            eval_accumulation_steps=64,
            save_steps=50,
            eval_steps=10,
            data_split_percent=[1, 1, 1],
            num_dialogs=[10, 5, 1],
            quantization=True,
            num_turns=26,
            should_add_schema=True,
            should_add_user_actions=True,
            should_add_service_results=True,
            train_domain_settings="seen",
            dev_domain_settings="all",
            test_domain_settings=["all"],
            context_type="nlg",
            prompt_type="default",
        )
    )
    tt.run()


@hydra.main(config_path="../config/t5_trainer/", config_name="t5_trainer")
# @hydra.main(config_path="../config/t5_trainer/", config_name="t5_inference")
def hydra_start(cfg: DictConfig) -> None:
    t5tod = T5Tod(cfg)
    t5tod.run()
    # sys.stdout.close()


if __name__ == "__main__":
    # deepspeed.init_distributed()
    hydra_start()
    # wandb.finish()
    # old_main()
    print(os.getcwd())
