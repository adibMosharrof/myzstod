import os
import sys
import uuid


sys.path.insert(0, os.path.abspath("./src"))
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
)
from datetime import datetime
import utils
from tqdm import tqdm
import numpy as np
import evaluate
import logging


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
        target_max_len = self.cfg.max_token_len - self.cfg.prompt_len
        for item in batch:
            context_tokens = self.my_tokenize(item.context)
            schema_tokens = self.my_tokenize(item.schema)
            context_unused_len = (
                self.cfg.prompt_len
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
    def __init__(self, cfg):
        self.cfg = cfg
        base_out_dir = self.cfg.project_root / "playground" / "t5_tod_out"
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        self.cfg.out_dir = base_out_dir / date_str / time_str
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.cfg.out_dir)
        log_file = self.cfg.out_dir / "t5_tod.log"
        logging.basicConfig(filename=log_file, level=logging.INFO, encoding="utf-8")

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

    def run(self):
        accelerator = Accelerator()
        torch.manual_seed(420)

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        if self.cfg.model_path:
            model = T5ForConditionalGeneration.from_pretrained(
                self.cfg.project_root / self.cfg.model_path
            ).cuda()
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

        training_args = TrainingArguments(
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
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.dm.tod_train_collate,
        )
        if not self.cfg.model_path:
            trainer.train()
            model.save_pretrained(self.cfg.out_dir)

        test_dl = DataLoader(
            test_dataset,
            batch_size=self.cfg.test_batch_size,
            collate_fn=self.dm.tod_train_collate,
            pin_memory=True,
            num_workers=8,
        )

        _ = model.eval()
        print("starting inference")

        all_labels, all_preds = [], []
        test_dl = accelerator.prepare(test_dl)
        for batch in tqdm(test_dl):
            max_gen_len = self.cfg.max_token_len - self.cfg.prompt_len
            sample_outputs = model.generate(
                inputs=batch.input_ids.to(accelerator.device),
                attention_mask=batch.attention_mask.to(accelerator.device),
                do_sample=False,
                max_length=max_gen_len,
            )
            out_padded = self.pad_gen_to_max_len(sample_outputs, max_gen_len, tokenizer)
            sample_outputs, labels = accelerator.gather_for_metrics(
                (out_padded, batch.labels)
            )
            # decode the predicted tokens into texts
            pred_text = tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
            target_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_labels.append(target_text)
            all_preds.append(pred_text)

        concat_labels = np.concatenate(all_labels, axis=0)
        concat_preds = np.concatenate(all_preds, axis=0)
        df = pd.DataFrame(
            {
                "target_text": concat_labels,
                "pred_text": concat_preds,
            }
        )
        out_path = self.cfg.out_dir / "t5_tod.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        google_bleu = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        gleu_labels = np.expand_dims(concat_labels, axis=1)
        result = google_bleu.compute(predictions=concat_preds, references=gleu_labels)
        score_str = f"GLEU score: {result['google_bleu']}"
        logging.info(score_str)
        print(score_str)


if __name__ == "__main__":
    tt = T5Tod(
        DotMap(
            # csv_file="nlg_data.csv",
            csv_file="v0_context_type_nlg_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_False_turns_10_service_results_True_dialogs_5_domain_setting_all_train_domains_1.0.csv",
            separate_dev_test=True,
            project_root=Path("/projects/bbyl/amosharrof/ZSToD"),
            tokenizer_name="adibm/sgd-flan-t5-nlg-tokenizer",
            model_name="google/flan-t5-large",
            # model_path="playground/t5_tod_out/2023-10-27/00-42-59",
            # model_path="outputs/2023-10-25/11-49-15/results/pretrain",
            model_path="",
            max_token_len=1024,
            prompt_len=750,
            train_batch_size=4,
            eval_batch_size=20,
            test_batch_size=50,
            epochs=20,
            gradient_accumulation_steps=64,
            eval_accumulation_steps=64,
            save_steps=50,
            eval_steps=25,
            data_split_percent=[1, 1, 0.5],
        )
    )
    tt.run()
