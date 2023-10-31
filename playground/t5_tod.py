import os
import sys
import uuid

from omegaconf import DictConfig

sys.path.insert(0, os.path.abspath("./src"))
from my_enums import Steps
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


class T5DataModule:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

    def my_tokenize(self, text: str):
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        return tokens.to(dtype=torch.int32)[0]

    def tod_train_collate(self, batch: list[TodTurnCsvRow]):
        all_input_tokens = []
        all_labels = []
        all_attention_masks = []
        prompt_text = "\n".join(
            [
                "Instructions: Given the Dialog History and the Dialog Schemas, please generate the system response.\n",
                "Dialog History\n",
            ]
        )
        prompt_tokens = self.my_tokenize(prompt_text)
        schema_prompt_tokens = self.my_tokenize("\n\nDialog Schemas\n")
        target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
        for item in batch:
            context_tokens = self.my_tokenize(item.context)
            schema_tokens = self.my_tokenize(item.schema)
            context_unused_len = (
                self.cfg.test_prompt_max_len
                - len(prompt_tokens)
                - len(context_tokens)
                - len(schema_prompt_tokens)
                - len(schema_tokens)
            )
            if context_unused_len < 0:
                context_tokens = context_tokens[context_unused_len * -1 :]
                context_unused_len = 0
            pad = torch.full([context_unused_len], self.tokenizer.pad_token_id)
            input_tokens = torch.cat(
                [
                    prompt_tokens,
                    context_tokens,
                    schema_prompt_tokens,
                    schema_tokens,
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
            self.cfg.project_root / "playground" / "data" / "train" / self.cfg.csv_file
        )
        val_fp = (
            self.cfg.project_root / "playground" / "data" / "dev" / self.cfg.csv_file
        )
        test_fp = (
            self.cfg.project_root / "playground" / "data" / "test" / self.cfg.csv_file
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
        fp = self.cfg.project_root / "playground" / "data" / self.cfg.csv_file
        # a = self.get_dms()
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
        # base_out_dir = self.cfg.project_root / "playground" / "t5_tod_out"
        # date_str = datetime.now().strftime("%Y-%m-%d")
        # time_str = datetime.now().strftime("%H-%M-%S")
        # self.cfg.out_dir = base_out_dir / date_str / time_str
        # self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        # os.chdir(self.cfg.out_dir)
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
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, load_in_8bit=True)
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
        # tokenizer.add_tokens(["<|user|>", "<|system|>"])
        if self.cfg.model_path:
            model = T5ForConditionalGeneration.from_pretrained(
                self.cfg.project_root / self.cfg.model_path
            ).cuda()
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
            train_dataset, val_dataset, test_dataset = self.dm.load_data()

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
            dataloader_num_workers=1,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            eval_accumulation_steps=self.cfg.eval_accumulation_steps,
            learning_rate=1e-3,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.dm.tod_train_collate,
        )
        if not self.cfg.model_path:
            trainer.train()
            # trainer.save_model()
            # model.save_pretrained(self.cfg.out_dir)
            if accelerator.is_main_process:
                trainer.model.save_pretrained(self.cfg.out_dir)
            accelerator.wait_for_everyone()

        test_dl = DataLoader(
            test_dataset,
            batch_size=self.cfg.test_batch_size,
            collate_fn=self.dm.tod_train_collate,
            pin_memory=True,
            num_workers=8,
        )

        _ = model.eval()
        print("starting inference")

        inf_logger = InferenceLogger(self.cfg.out_dir / "t5_tod.csv", tokenizer)
        if self.cfg.quantization:
            config = PeftConfig.from_pretrained(self.cfg.out_dir)
            device_map = {"": accelerator.device}
            model = AutoModelForSeq2SeqLM.from_pretrained(
                config.base_model_name_or_path,
                load_in_8bit=False,
                device_map=device_map,
            )
            model.resize_token_embeddings(len(tokenizer))
            model = PeftModel.from_pretrained(
                model, self.cfg.out_dir, device_map=device_map
            )
            model.eval()

        test_dl = accelerator.prepare(test_dl)
        for batch in tqdm(test_dl):
            max_gen_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
            sample_outputs = model.generate(
                inputs=batch.input_ids.to(accelerator.device),
                attention_mask=batch.attention_mask.to(accelerator.device),
                do_sample=False,
                max_length=max_gen_len,
            )
            out_padded = self.pad_gen_to_max_len(sample_outputs, max_gen_len, tokenizer)
            padded_outputs, label_tokens, input_tokens = accelerator.gather_for_metrics(
                (out_padded, batch.labels, batch.input_ids)
            )
            # decode the predicted tokens into texts
            inf_logger.add_batch(input_tokens, label_tokens, padded_outputs)

        inf_logger.write_csv()
        metric_manager = NlgMetricManager(self.logger)
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
            data_split_percent=[0.1, 0.5, 0.1],
            quantization=True,
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
