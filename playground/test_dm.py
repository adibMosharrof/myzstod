from dotmap import DotMap
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))

from datamodules.data_collators.collator_factory import CollatorFactory
from prompts.nlg_prompt_manager import NlgPromptFactory


from configs.dm_config import DataModuleConfig
from datamodules.tod_datamodulev2 import TodDataModuleV2
from tod.turns.zs_tod_turn import TodTurnCsvRowFactory


from sgd_dstc8_data_model.dstc_dataclasses import get_schemas
from utilities.tokenizer_utilities import TokenizerUtilities
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
        tokenizer = TokenizerUtilities.get_tokenizer(
            model_name=self.cfg.model_name, context_type=self.cfg.context_type
        )
        tokenizer.model_max_length = 1024
        schemas = {}
        steps = Steps.list()
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in steps]:
            schemas.update(d)

        prompt_cls = NlgPromptFactory.get_handler(
            self.cfg.prompt_type, self.cfg.model_type.context_type
        )
        collator = CollatorFactory.create_collator(
            model_name=self.cfg.model_name,
            context_type=self.cfg.context_type,
            tokenizer=tokenizer,
            prompt_cls=prompt_cls,
            max_token_len=self.cfg.max_token_len,
            test_prompt_max_len=self.cfg.test_prompt_max_len,
            schema_max_len=self.cfg.get("schema_max_len", 350),
        )

        tod_turn_row_cls = TodTurnCsvRowFactory.get_handler(self.cfg)
        dm = TodDataModuleV2(
            DataModuleConfig(tokenizer=tokenizer, **self.cfg),
            tod_turn_row_cls=tod_turn_row_cls,
            schemas=schemas,
        )
        dm.setup()
        train_dataset, val_dataset, test_datasets = (
            dm.datasets["train"],
            dm.datasets["dev"],
            dm.datasets["test"],
        )

        train_dl = DataLoader(
            train_dataset,
            batch_size=self.cfg.train_batch_size,
            collate_fn=collator.tod_train_collate,
            pin_memory=True,
            num_workers=1,
        )
        train_dl = accelerator.prepare(train_dl)
        print("train_dl")
        for batch in tqdm(train_dl):
            max_gen_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len

        val_dl = DataLoader(
            val_dataset,
            batch_size=self.cfg.eval_batch_size,
            collate_fn=collator.tod_train_collate,
            pin_memory=True,
            num_workers=1,
        )
        val_dl = accelerator.prepare(val_dl)
        print("val_dl")
        for batch in tqdm(val_dl):
            max_gen_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len

        print("test_dl")
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
                collate_fn=collator.tod_test_collate,
                pin_memory=True,
                num_workers=1,
            )
            test_dl = accelerator.prepare(test_dl)
            for batch in tqdm(test_dl):
                max_gen_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len


if __name__ == "__main__":
    cfg = DotMap(
        # csv_file="nlg_data.csv",
        # csv_file="v0_context_type_nlg_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_False_turns_26_service_results_True_dialogs_1_domain_setting_all_train_domains_1.0.csv",
        separate_dev_test=False,
        # project_root=Path("/projects/bbyl/amosharrof/ZSToD"),
        project_root=Path("/mounts/u-amo-d1/adibm-data/projects/ZSToD/"),
        # data_prep_out_root="processed_data/simple_tod",
        # raw_data_root="data/dstc8-schema-guided-dialogue/",
        # data_prep_out_root="processed_data/sgd_x/v5",
        # raw_data_root="data/dstc8-schema-guided-dialogue/sgd_x/data/v5",
        data_prep_out_root="processed_data/bitod",
        raw_data_root="data/bitod/",
        dataset_name="bitod",
        # tokenizer_name="adibm/sgd-flan-t5-nlg-tokenizer",
        # model_name="google/flan-t5-large",
        model_name="gpt2",
        # model_path="playground/t5_tod_out/2023-11-28/03-15-51",
        # model_path="outputs/2023-10-25/11-49-15/results/pretrain",
        model_path="",
        max_token_len=1024,
        test_prompt_max_len=640,
        train_batch_size=50,
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
        num_dialogs=[-1, -1, -1],
        # num_dialogs=[1, 1, 34],
        quantization=False,
        num_turns=26,
        # should_add_schema=True,
        should_add_schema=False,
        should_add_user_actions=False,
        should_add_service_results=True,
        train_domain_settings="all",
        is_scale_grad=False,
        # train_domain_settings=["Banks_1", "Hotels_2"],
        dev_domain_settings=["all"],
        # dev_domain_settings=["Banks_2", "Hotels_1"],
        # test_domain_settings=[["all"], ["seen"], ["unseen"]],
        test_domain_settings=[["all"]],
        # test_domain_settings=[["Hotels_4"], ["Restaurants_2"]],
        # context_type="nlg",
        # context_type="bitod",
        prompt_type="default",
        # prompt_type="multi_domain",
        # overwrite=[1, 1, 1],
        data_prep_multi_process=True,
        # data_prep_multi_process=False,
        # context_type="zstod_api_call",
        context_type="bitod_soloist_api_call",
    )

    cfg.model_type = DotMap(context_type="bitod_soloist_api_call")
    # cfg.datasets = DotMap(
    #     bitod=DotMap(
    #         raw_data_root="data/bitod",
    #         data_prep_out_root="processed_data/bitod",
    #         dataset_name="bitod",
    #     )
    # )
    test_dm = TestDM(cfg)
    test_dm.run()
