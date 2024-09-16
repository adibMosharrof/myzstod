from collections import Counter, defaultdict
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath("./src"))
from dotmap import DotMap
import numpy as np
import pandas as pd
from tqdm import tqdm

from tod.turns.zs_tod_turn import TodTurnMultiTaskCsvRow
from tod_datamodules import TodDataModule
from configs.dm_config import DataModuleConfig

from my_enums import Steps, MultiTaskNames
import utils
from dataclasses import make_dataclass
import seaborn as sns
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Union

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig, PeftModelForCausalLM
import utils
import accelerate
from accelerate import PartialState, Accelerator
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist


class TokenLengths:
    token_types = ["context", "target"]

    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.tokenizer = utils.get_tokenizer(cfg.model_name)
        self.cfg.multi_tasks = (
            MultiTaskNames.get_multi_task_names(cfg.multi_tasks)
            if cfg.is_multi_task
            else cfg.multi_tasks
        )
        self.context_lens = []
        self.target_lens = []
        self.corpus = []

    def get_dms(self):
        steps = Steps.list()
        if self.cfg.is_multi_task:
            return [
                TodDataModule(
                    DataModuleConfig(**self.cfg),
                    steps=steps,
                    tod_turn_row_cls=TodTurnMultiTaskCsvRow,
                    task_name=task_name,
                )
                for task_name in self.cfg.multi_tasks
            ]
        return [
            TodDataModule(
                DataModuleConfig(**self.cfg),
                steps=steps,
            )
        ]

    def process_dm(self, dm):
        for data in dm:
            context_tokens = self.cfg.tokenizer.tokenize(data.context + data.schema)
            target_tokens = self.cfg.tokenizer.tokenize(data.target)
            self.context_lens.append(len(context_tokens))
            self.target_lens.append(len(target_tokens))

    def get_stats(self):
        out = []
        freq = []
        for key, val in zip(self.token_types, [self.context_lens, self.target_lens]):
            out.append(TokenStats(key, "mean", np.round(np.mean(val), 2)))
            out.append(TokenStats(key, "max", np.max(val)))
            out.append(TokenStats(key, "median", np.round(np.median(val), 2)))
            for v in val:
                freq.append(TokenStats(key, "freq", v))
        return out, freq

    def plot_graphs(self, mmm, freq):
        sns.set_palette("muted")
        df = pd.DataFrame(mmm)
        plt.figure()
        g = sns.catplot(x="token_type", y="value", hue="stat_name", data=df, kind="bar")
        ax = g.facet_axis(0, 0)
        for c in ax.containers:
            labels = [v.get_height() for v in c]
            ax.bar_label(c, labels=labels, label_type="edge")
        g.figure.savefig(
            f"data_exploration/token_stats/mt_{self.cfg.is_multi_task}_token_stats_{'_'.join(map(str,self.cfg.num_dialogs))}.png"
        )

        plt.figure()
        freq_df = pd.DataFrame(freq)
        g = sns.displot(
            x="value", hue="token_type", data=freq_df, multiple="stack", alpha=0.5
        )
        g.figure.savefig(
            f"data_exploration/token_stats/mt_{self.cfg.is_multi_task}_freq_{'_'.join(map(str,self.cfg.num_dialogs))}.png"
        )

    def train_tokenizer(self):
        new_tokenizer = self.cfg.tokenizer.train_new_from_iterator(self.corpus, 52000)
        self.cfg.tokenizer = new_tokenizer
        new_tokenizer.save_pretrained("tokenizer")
        # new_tokenizer.push_to_hub("adibm/sgd-llama-tokenizer")
        new_tokenizer.push_to_hub("adibm/sgd-opt-tokenizer")

    def create_corpus(self, dm):
        for data in dm:
            corpus_text = "".join([data.context, data.schema, data.target])
            self.corpus.append(corpus_text)

    def test_inf(self, dm):
        # model_name = "huggyllama/llama-7b"
        model_name = "facebook/opt-350m"
        tokenizer_name = "adibm/sgd-llama-tokenizer"
        adapter_name = "default"
        # m_path = "outputs/2023-09-15/15-00-37/results/pretrain"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = utils.get_8bit_model(model_name, is_inference=False)
        model.eval()
        model.get_memory_footprint()
        # distributed_state = PartialState()

        ds = dm.datasets[Steps.TRAIN.value][:10]
        train_loader = DataLoader(
            dataset=ds,
            batch_size=2,
            collate_fn=dm.my_test_collate,
        )
        accelerator = Accelerator()
        train_loader = accelerator.prepare(train_loader)
        outputs = []
        for batch in train_loader:
            inp = batch.turn_ids.to(accelerator.device)
            gen = model.generate(inputs=inp, max_length=5)
            out = accelerator.gather_for_metrics(gen)
            outputs.append(out)
        outputs_reshaped = torch.cat(outputs, 0)
        out_str = tokenizer.batch_decode(outputs_reshaped, skip_special_tokens=True)
        print(out_str)
        return
        outputs = []
        for batches in utils.grouper(train_loader, world_size):
            with distributed_state.split_between_processes(batches) as prompt:
                batch_data = prompt[0]
                if not batch_data:
                    continue
                print(batch_data.turn_ids)
                inp = tokenizer(batch_data.turn_ids, return_tensors="pt").to("cuda")
                # inp = [p.turn_ids for p in prompt if p is not None]
                # tokens = tokenizer(inp, return_tensors="pt").to("cuda")
                # tokens = tokenizer(prompt.turn_id, return_tensors="pt").to("cuda")
                gen = model.generate(
                    inputs=inp.input_ids,
                    # inputs=prompt,
                    # inputs=inp,
                    max_length=5,
                )
                # dist.all_gather(outputs, gen)
                outputs.append(gen)
                # outputs[accelerator.process_index].append(gen)
            # outputs = model.generate(
            #         inputs=input_ids.input_ids,
            #         max_length=10,
            #     )

        accelerator.wait_for_everyone()
        outputs_reshaped = torch.cat(outputs, 0)
        out_str = tokenizer.batch_decode(outputs_reshaped, skip_special_tokens=True)
        print(out_str)
        # for i in outputs:
        #     print(tokenizer.batch_decode(i, skip_special_tokens=True))

    def run(self):
        dms = self.get_dms()
        # self.test_inf(dms[0])
        # self.test_inf(None)
        # return
        for step in Steps.list():
            for step_dm in dms:
                dm = step_dm.datasets[step]
                if step == Steps.TEST.value:
                    [self.create_corpus(d) for d in dm]
                else:
                    self.create_corpus(dm)

        self.train_tokenizer()
        for step in Steps.list():
            for step_dm in dms:
                dm = step_dm.datasets[step]
                if step == Steps.TEST.value:
                    [self.process_dm(d) for d in dm]
                else:
                    self.process_dm(dm)
        mmm, freq = self.get_stats()
        self.plot_graphs(mmm, freq)

    def mt_gen(self):
        accelerator = Accelerator()

        # a = torch.hstack(
        #     [
        #         torch.full([5, 3], 1, device=accelerator.device),
        #         torch.full([5, 7], 2, device=accelerator.device),
        #     ]
        # )
        # b = torch.hstack(
        #     [
        #         torch.full([5, 3], 3, device=accelerator.device),
        #         torch.full([5, 7], 4, device=accelerator.device),
        #     ]
        # )
        # c = torch.hstack(
        #     [
        #         torch.full([5, 3], 5, device=accelerator.device),
        #         torch.full([5, 7], 6, device=accelerator.device),
        #     ]
        # )
        # gen = [a, b, c]
        gen = torch.arange(150, device=accelerator.device).reshape(3, 5, 10)
        no_c = []
        for g in gen:
            out = g[:, 3:]
            no_c.append(out)
        gen_cat = torch.hstack([*no_c])
        all_gen_cat = accelerator.gather_for_metrics(gen_cat)

        if accelerator.is_main_process:
            testing = all_gen_cat * -1
            print(testing[0][0])
        accelerator.wait_for_everyone()
        print(f"process {accelerator.process_index}\n")
        x = 1


@dataclass
class TokenStats:
    token_type: str
    stat_name: str
    value: Union[float, list[float]]


if __name__ == "__main__":
    mdd = TokenLengths(
        DotMap(
            raw_data_root=Path("data/dstc8-schema-guided-dialogue/"),
            data_prep_out_root="data/processed_data/",
            project_root=Path("/mounts/u-amo-d1/adibm-data/projects/ZSToD"),
            num_dialogs=[127, 20, 34],
            # num_dialogs=[1, 1, 1],
            data_split_percent=[1, 1, 1],
            max_token_len=1024,
            test_prompt_max_len=750,
            # model_name="huggyllama/llama-7b",
            model_name="facebook/opt-2.7b",
            # out_file_path="data_exploration/token_lengths",
            should_add_schema=True,
            should_add_sys_actions=False,
            should_add_user_actions=True,
            should_add_service_results=True,
            is_multi_task=False,
            multi_tasks=["dsts", "actions", "nlg"],
            train_domain_settings=["seen"],
            dev_domain_settings=["all"],
            test_domain_settings=[["all"]],
            # test_domain_settings=[["all"], ["seen"], ["unseen"]],
        )
    )
    # mdd.run()
    mdd.mt_gen()
