from dotmap import DotMap
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas
from transformers import AutoTokenizer
from t5_datamodule import T5DataModule
from my_enums import Steps
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path


class TestDM:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.cfg.raw_data_root = self.cfg.project_root / self.cfg.raw_data_root

    def run(self):
        accelerator = Accelerator()
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_name or self.cfg.model_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<SYSTEM>", "<USER>"]}
        )
        tokenizer.model_max_length = 1024
        schemas = {}
        steps = Steps.list()
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in steps]:
            schemas.update(d)
        dm = T5DataModule(self.cfg, tokenizer, schemas)
        train_dataset, val_dataset, test_datasets = dm.load_data()
        for test_dataset, domain_names_list in zip(
            test_datasets, self.cfg.test_domain_settings
        ):
            domain_names = ",".join(domain_names_list)
            if not len(test_dataset):
                print(f"No data for {domain_names}")
                continue
            print(f"testing {domain_names}")
            test_dl = DataLoader(
                test_dataset,
                batch_size=self.cfg.test_batch_size,
                collate_fn=dm.tod_test_collate,
                pin_memory=True,
                num_workers=1,
            )
            test_dl = accelerator.prepare(test_dl)
            for batch in tqdm(test_dl):
                max_gen_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len


if __name__ == "__main__":
    test_dm = TestDM(
        DotMap(
            # csv_file="nlg_data.csv",
            # csv_file="v0_context_type_nlg_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_False_turns_26_service_results_True_dialogs_1_domain_setting_all_train_domains_1.0.csv",
            separate_dev_test=False,
            # project_root=Path("/projects/bbyl/amosharrof/ZSToD"),
            project_root=Path("/mounts/u-amo-d1/adibm-data/projects/ZSToD/"),
            data_prep_out_root="processed_data/ketod",
            # raw_data_root="data/dstc8-schema-guided-dialogue/",
            raw_data_root="data/ketod/",
            # tokenizer_name="adibm/sgd-flan-t5-nlg-tokenizer",
            model_name="google/flan-t5-base",
            model_path="playground/t5_tod_out/2023-11-28/03-15-51",
            # model_path="outputs/2023-10-25/11-49-15/results/pretrain",
            # model_path="",
            max_token_len=1024,
            test_prompt_max_len=850,
            train_batch_size=10,
            eval_batch_size=20,
            test_batch_size=60,
            # epochs=3,
            epochs=1,
            gradient_accumulation_steps=64,
            eval_accumulation_steps=64,
            save_steps=50,
            # eval_steps=1,
            eval_steps=10,
            # data_split_percent=[0.1, 1, 0.5],
            data_split_percent=[1, 1, 1],
            # num_dialogs=[127, 20, 34],
            # num_dialogs=[1, 1, 1],
            num_dialogs=[10, 10, 10],
            quantization=True,
            num_turns=26,
            should_add_schema=True,
            should_add_user_actions=True,
            should_add_service_results=True,
            train_domain_settings="seen",
            is_scale_grad=False,
            # train_domain_settings=["Banks_1", "Hotels_2"],
            dev_domain_settings=["all"],
            # dev_domain_settings=["Banks_2", "Hotels_1"],
            # test_domain_settings=[["all"], ["seen"], ["unseen"]],
            test_domain_settings=[["all"]],
            # test_domain_settings=[["Hotels_4"], ["Restaurants_2"]],
            # context_type="nlg",
            # context_type="nlg_api_call",
            context_type="ketod_api_call",
            prompt_type="default",
            # prompt_type="multi_domain",
            # overwrite=[1, 1, 1],
            # data_prep_multi_process=False,
        )
    )
    test_dm.run()
