import os
import sys
import uuid

from omegaconf import DictConfig


sys.path.insert(0, os.path.abspath("./src"))
from my_enums import Steps
from prompts.nlg_prompt_manager import NlgPromptFactory
from tod_datamodules import TodDataModule
from configs.dm_config import DataModuleConfig
import hydra
from logger.inference_logger import InferenceLogger
from metric_managers.nlg_metric_manager import NlgMetricManager
from tod.turns.zs_tod_turn import TodTurnCsvRow
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


class T5DataModule:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.nlg_prompt_cls = NlgPromptFactory.get_handler(cfg.prompt_type)

    def my_tokenize(self, text: str, max_len: int = None):
        tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=max_len)
        return tokens.to(dtype=torch.int32)[0]

    def trim_dialog_history(self, item: TodTurnCsvRow, trim_len: int):
        dialog_history_tokens = self.my_tokenize(item.context)
        trimmed_history_tokens = dialog_history_tokens[trim_len + 5 :]
        trimmed_history_text = self.tokenizer.decode(trimmed_history_tokens)
        context_text = self.nlg_prompt_cls.get_prompt(
            item.domains, item.schema, trimmed_history_text
        )
        context_tokens = self.my_tokenize(context_text)
        return context_tokens

    def tod_train_collate(self, batch: list[TodTurnCsvRow]):
        all_input_tokens = []
        all_labels = []
        all_attention_masks = []

        target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
        for item in batch:
            context_text = self.nlg_prompt_cls.get_prompt(
                item.domains, item.schema, item.context
            )
            context_tokens = self.my_tokenize(context_text)
            context_unused_len = self.cfg.test_prompt_max_len - len(context_tokens)
            if context_unused_len < 0:
                context_tokens = self.trim_dialog_history(item, -context_unused_len)
                context_unused_len = 0
            pad = torch.full([context_unused_len], self.tokenizer.pad_token_id)
            input_tokens = torch.cat(
                [
                    context_tokens,
                    pad,
                ]
            )
            attention_mask = input_tokens.ne(self.tokenizer.pad_token_id).to(
                torch.int32
            )

            target_tokens = self.my_tokenize(item.target)
            target_unused_len = target_max_len - len(target_tokens)
            label = torch.cat(
                [
                    target_tokens,
                    torch.full([target_unused_len], self.tokenizer.pad_token_id),
                    torch.full([1], self.tokenizer.eos_token_id),
                ]
            )
            all_input_tokens.append(input_tokens)
            all_attention_masks.append(attention_mask)
            all_labels.append(label)
        return DotMap(
            {
                "input_ids": torch.stack(all_input_tokens),
                "labels": torch.stack(all_labels),
                "attention_mask": torch.stack(all_attention_masks),
            }
        )

    def get_data_by_split_percent(
        self, data: list[TodTurnCsvRow], split_percent: float
    ):
        return data[: int(len(data) * split_percent)]

    def get_dms(self):
        steps = Steps.list()

        return [
            TodDataModule(
                DataModuleConfig(tokenizer=self.tokenizer, **self.cfg),
                steps=steps,
            )
        ]

    def load_data_from_files(self):
        train_fp = (
            self.cfg.project_root
            / "playground"
            / "data"
            / "train"
            / self.cfg.train_csv_file
        )
        val_fp = (
            self.cfg.project_root
            / "playground"
            / "data"
            / "dev"
            / self.cfg.dev_csv_file
        )
        test_fp = (
            self.cfg.project_root
            / "playground"
            / "data"
            / "test"
            / self.cfg.test_csv_file
        )
        train_data = utils.read_csv_dataclass(train_fp, TodTurnCsvRow)
        val_data = utils.read_csv_dataclass(val_fp, TodTurnCsvRow)
        test_data = utils.read_csv_dataclass(test_fp, TodTurnCsvRow)
        datasets = [
            SimpleTodDataSet(self.get_data_by_split_percent(data, split))
            for data, split in zip(
                [train_data, val_data, test_data], self.cfg.data_split_percent
            )
        ]
        return (*datasets,)

    def load_data(self):
        tod_dms = self.get_dms()[0].datasets
        return tod_dms["train"], tod_dms["dev"], tod_dms["test"]
        fp = self.cfg.project_root / "playground" / "data" / self.cfg.csv_file
        data = utils.read_csv_dataclass(fp, TodTurnCsvRow)
        df = pd.DataFrame([vars(d) for d in data])
        # df = pd.read_csv(fp, encoding="ISO-8859-1", header=None)

        df = df.sample(1200, random_state=420)

        # divide into test and train
        train_df = df.sample(frac=0.8, random_state=420)
        rest_df = df.drop(train_df.index)
        val_df = rest_df.sample(frac=0.5, random_state=420)
        test_df = rest_df.drop(val_df.index)
        datasets = [
            SimpleTodDataSet(
                df.apply(lambda row: TodTurnCsvRow(**row), axis=1).to_list()
            )
            for df in [train_df, val_df, test_df]
        ]
        return (*datasets,)


class T5Tod:
    def __init__(self, cfg, **kwargs):
        self.cfg = DotMap(dict(cfg))
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

    def get_model(self, model_name: str, tokenizer: AutoTokenizer):
        model_path = model_name
        if self.cfg.model_path:
            model_path = self.cfg.project_root / self.cfg.model_path
        if not self.cfg.quantization:
            model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
            return model
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForSeq2SeqLM.from_pretrained(
            # model_path, load_in_8bit=True, torch_dtype=torch.bfloat16
            model_path,
            load_in_8bit=False,
            torch_dtype=dtype
            # model_path, load_in_8bit=False
        )
        model.resize_token_embeddings(len(tokenizer))
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            modules_to_save=utils.get_modules_to_save(model_name),
        )
        model = get_peft_model(model, lora_config)
        return model

    def run(self):
        accelerator = Accelerator()
        torch.manual_seed(420)

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_name or self.cfg.model_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        model = None
        # tokenizer.add_tokens(["<|user|>", "<|system|>"])
        if self.cfg.model_path:
            model_out_dir = str(self.cfg.project_root / self.cfg.model_path)
        elif self.cfg.quantization:
            model = self.get_model(self.cfg.model_name, tokenizer)
        else:
            model = T5ForConditionalGeneration.from_pretrained(
                self.cfg.model_name
            ).cuda()
            model.resize_token_embeddings(len(tokenizer))

        self.dm = T5DataModule(self.cfg, tokenizer)
        if self.cfg.separate_dev_test:
            train_dataset, val_dataset, test_dataset = self.dm.load_data_from_files()
        else:
            train_dataset, val_dataset, test_datasets = self.dm.load_data()

        if not self.cfg.model_path:
            training_args = Seq2SeqTrainingArguments(
                output_dir=str(self.cfg.out_dir),
                num_train_epochs=self.cfg.epochs,
                logging_steps=10,
                save_total_limit=5,
                save_steps=self.cfg.save_steps,
                eval_steps=self.cfg.eval_steps,
                load_best_model_at_end=True,
                save_strategy=IntervalStrategy.STEPS,
                evaluation_strategy=IntervalStrategy.STEPS,
                per_device_train_batch_size=self.cfg.train_batch_size,
                per_device_eval_batch_size=self.cfg.eval_batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                dataloader_drop_last=True,
                dataloader_num_workers=8,
                gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
                eval_accumulation_steps=self.cfg.eval_accumulation_steps,
                learning_rate=1e-3,
                # bf16_full_eval=True,
                # bf16=True,
                fp16=True,
                fp16_full_eval=True,
                # gradient_checkpointing=False,
                # ddp_find_unused_parameters=False,
            )
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=self.dm.tod_train_collate,
            )
            # model.gradient_checkpointing_disable()
            # with accelerator.no_sync(model):
            # trainer.train()
            trainer.train()
            # trainer.save_model()
            # model.save_pretrained(self.cfg.out_dir)
            if accelerator.is_main_process:
                trainer.model.save_pretrained(self.cfg.out_dir)
            accelerator.wait_for_everyone()
            model_out_dir = self.cfg.out_dir

        if not self.cfg.model_path:
            _ = model.eval()
        print("starting inference")
        metric_manager = NlgMetricManager(self.logger)
        inf_logger = InferenceLogger(tokenizer, metric_manager)
        if self.cfg.quantization:
            config = PeftConfig.from_pretrained(model_out_dir)
            device_map = {"": accelerator.device}
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path,
                load_in_8bit=False,
                device_map=device_map,
            )
            model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(
                model, model_out_dir, device_map=device_map
            )
            model.eval()

        for test_dataset, domain_names_list in zip(
            test_datasets, self.cfg.test_domain_settings
        ):
            domain_names = ",".join(domain_names_list)
            if not len(test_dataset):
                print(f"No data for {domain_names}")
                continue
            test_dl = DataLoader(
                test_dataset,
                batch_size=self.cfg.test_batch_size,
                collate_fn=self.dm.tod_train_collate,
                pin_memory=True,
                num_workers=8,
            )
            test_dl = accelerator.prepare(test_dl)
            for batch in tqdm(test_dl):
                max_gen_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
                with torch.no_grad():
                    sample_outputs = model.generate(
                        inputs=batch.input_ids.to(accelerator.device),
                        attention_mask=batch.attention_mask.to(accelerator.device),
                        # do_sample=False,
                        max_length=max_gen_len,
                        # penalty_alpha=0.6,
                        do_sample=True,
                        top_k=50,
                        top_p=0.92,
                        num_return_sequences=1,
                    )
                out_padded = self.pad_gen_to_max_len(
                    sample_outputs, max_gen_len, tokenizer
                )
                (
                    padded_outputs,
                    label_tokens,
                    input_tokens,
                ) = accelerator.gather_for_metrics(
                    (out_padded, batch.labels, batch.input_ids)
                )
                # decode the predicted tokens into texts
                inf_logger.add_batch(input_tokens, label_tokens, padded_outputs)
            csv_path = self.cfg.out_dir / f"{domain_names}.csv"
            inf_logger.write_csv(csv_path)

            metric_manager.compute_metrics(inf_logger)


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
def hydra_start(cfg: DictConfig) -> None:
    print(os.getcwd())
    t5tod = T5Tod(cfg)
    t5tod.run()
    sys.stdout.close()


if __name__ == "__main__":
    # deepspeed.init_distributed()
    # hydra_start()
    # wandb.finish()
    old_main()
    print(os.getcwd())
