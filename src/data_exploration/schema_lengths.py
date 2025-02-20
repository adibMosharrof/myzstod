from pathlib import Path
from dotmap import DotMap
import os
import sys
from transformers import AutoTokenizer
import pandas as pd

sys.path.insert(0, os.path.abspath("./src"))


class SchemaLengths:

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def run(self):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        schema_lens = []
        for csv_file in self.cfg.csv_files:
            df = pd.read_csv(csv_file)
            # schema_lens.append(
            #     df["schema"].apply(lambda x: len(tokenizer(x)["input_ids"])).max()
            # )

            # max_row = df.loc[df["schema"].str.len().idxmax()]
            # schema_lens.append(len(tokenizer.encode(max_row["schema"])))
            max_len = df["schema"].apply(lambda x: len(tokenizer.encode(x))).max()
            schema_lens.append(max_len)
        print(max(schema_lens))
        a = 1


if __name__ == "__main__":
    astats = SchemaLengths(
        DotMap(
            csv_files=[
                "/u/amo-d0/grad/adibm/v0_context_type_gpt_pseudo_labels_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_False_sys_actions_False_turns_26_service_results_True_dialogs_127_domain_setting_all_train_domains_1.0.csv"
                # "processed_data/sgd_x/pl0/train/v0_context_type_gpt_pseudo_labels_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_False_sys_actions_False_turns_26_service_results_True_dialogs_127_domain_setting_all_train_domains_1.0.csv"
                # "processed_data/simple_tod/train/v0_context_type_nlg_api_call_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_False_turns_26_service_results_True_dialogs_127_domain_setting_all_train_domains_1.0.csv",
                # "processed_data/simple_tod/test/v0_context_type_gpt_api_call_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_False_turns_26_service_results_True_dialogs_34_domain_setting_unseen_train_domains_1.0.csv",
                # "processed_data/simple_tod/dev/v0_context_type_nlg_api_call_scale_grad_False_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_False_turns_26_service_results_True_dialogs_20_domain_setting_all_train_domains_1.0.csv",
            ]
        )
    )
    astats.run()
