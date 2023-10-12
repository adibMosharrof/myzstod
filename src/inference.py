import os
import re
from pathlib import Path
from typing import Tuple, Union
from dotmap import DotMap
import omegaconf
import wandb
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from tqdm import tqdm
from configs.inference_config import InferenceConfig
from configs.reconstruct_dialog_config import ReconstructDialogConfig
from metrics.intent_accuracy_metric import IntentAccuracyMetric
from metrics.response_metrics import ResponseMetric
from collections import Counter, defaultdict

from torchmetrics import MetricCollection
from metrics.goal_metric import GoalMetric, GoalMetricConfigFactory
from metrics.requested_slots_metric import RequestedSlotsMetric
from metrics.dstc_metrics import InformMetric, SuccessMetric, CombinedMetric
from multi_woz.multi_woz_schema import MultiWozSchema
from my_enums import GoalMetricConfigType, MultiTaskNames, SpecialTokens, Steps
from reconstruct_dialog import ReconstructDialog
import utils
from simple_tod_dataclasses import (
    InferenceRecords,
    TodTestDataBatch,
)
from sgd_dstc8_data_model.dstc_dataclasses import get_slot_categories
import os
from accelerate import Accelerator
import cProfile

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Inference:
    def __init__(
        self,
        cfg: InferenceConfig,
    ):
        self.cfg = cfg
        self._set_metrics()

    def test(self):
        self.cfg.logger.info(self.cfg.out_dir)
        target_start_txt = "".join(
            [
                SpecialTokens.begin_target,
                SpecialTokens.begin_dsts,
                SpecialTokens.begin_dst,
            ]
        )
        start_tokens = []
        metric_results = []

        test_dl_func = (
            self.cfg.datamodule.grouped_test_dataloader
            if self.cfg.test_num_turns_groups
            else self.cfg.datamodule.test_dataloader
        )

        for test_dataloader, domain_setting in test_dl_func():
            domains_str = utils.get_domain_setting_str(domain_setting)
            test_csv_out_data = []
            text_csv_out_path = f"simple_tod_dstc_predictions_{domains_str}_{self.cfg.num_turns}_dialogs_{self.cfg.num_test_dialogs}.csv"
            if not len(test_dataloader):
                self.cfg.logger.info(f"No data to test for {domains_str}")
                continue
            inf_records = InferenceRecords()
            test_dataloader = self.cfg.accelerator.prepare(test_dataloader)
            for curr_batch in tqdm(test_dataloader):
                (
                    targets_text,
                    pred_text_no_pad,
                    contexts,
                    dialog_ids,
                    turn_ids,
                ) = self.cfg.generation_handler.get_generation(
                    curr_batch,
                    self.cfg.max_token_len - self.cfg.test_prompt_max_len,
                    self.cfg.max_token_len,
                    self.cfg.test_prompt_max_len,
                    self.cfg.postprocess_generation,
                    self.cfg.accelerator,
                )
                # if self.cfg.accelerator.is_main_process:
                if not self.cfg.is_multi_task:
                    self.tod_metrics.update(
                        references=targets_text, predictions=pred_text_no_pad
                    )
                    self.combined_metrics.update(
                        references=targets_text, predictions=pred_text_no_pad
                    )
                inf_records.add(
                    pred_text_no_pad,
                    targets_text,
                    dialog_ids,
                    turn_ids,
                    contexts,
                )
                # self.cfg.accelerator.wait_for_everyone()
            inf_records.concat_data()
            test_csv_out_data = np.column_stack(
                [
                    inf_records.dialog_ids,
                    inf_records.turn_ids,
                    inf_records.contexts,
                    inf_records.refs,
                    inf_records.preds,
                ]
            )
            if self.cfg.is_multi_task:
                preds, refs = inf_records.get_data_for_multitask()
                self.tod_metrics.update(references=refs, predictions=preds)
                self.combined_metrics.update(references=refs, predictions=preds)
            # self.cfg.accelerator.wait_for_everyone()
            # if self.cfg.accelerator.is_main_process:
            # with self.cfg.accelerator.main_process_first():
            if self.cfg.accelerator.is_main_process:
                headers = ["dialog_id", "turn_id", "context", "target", "prediction"]
                try:
                    utils.write_csv(headers, test_csv_out_data, text_csv_out_path)
                except Exception as e:
                    print("Could not write csv file as output")
            self.cfg.logger.info(f"Testing {domains_str}")
            cols, values = self._print_metrics()
            metric_results.append([domains_str, cols, values])
            self.cfg.logger.info(str(self.cfg.out_dir))
            [
                self.tod_metrics[m].visualize(Path(self.cfg.predictions_log_dir))
                for m in self.tod_metrics
            ]
            # self.cfg.accelerator.wait_for_everyone()
        if len(metric_results):
            self.log_metrics_wandb(metric_results)
        # self.cfg.logger.info("Start token counts")
        # for token, count in sorted(Counter(start_tokens).items()):
        #     self.cfg.logger.info(f"{token}:{count}")
        # r = ReconstructDialog(ReconstructDialogConfig.from_inference_config(self.cfg))
        # r.run()

    def run(self):
        print("begin inference")
        self.test()
        print("end inference")
        print("-" * 80)
        print("output_dir: ")
        print(os.getcwd())

    def log_metrics_wandb(self, metric_results):
        df = pd.DataFrame(
            columns=["Domains"] + metric_results[0][1],
            data=[[r[0]] + r[2] for r in metric_results],
        )
        wandb.log({"metrics": wandb.Table(dataframe=df)})

    def _print_metrics(self) -> Tuple[list[str], list[str]]:
        tod_metrics_str = [str(self.tod_metrics[m]) for m in self.tod_metrics]
        combined_metrics_str = [
            str(self.combined_metrics[m]) for m in self.combined_metrics
        ]
        all_metric_str = "\n".join(
            np.concatenate([tod_metrics_str, combined_metrics_str])
        )
        metric_strs = all_metric_str.split("\n")
        cols = []
        header_sep = []
        values = []
        for metric_str in metric_strs:
            if not metric_str:
                continue
            col, value = metric_str.split(":")
            cols.append(col)
            header_sep.append("-")
            values.append(value)
        self.cfg.logger.info(f"|{'|'.join(cols)}|")
        self.cfg.logger.info(f"|{'|'.join(header_sep)}|")
        self.cfg.logger.info(f"|{'|'.join(values)}|")
        return cols, values

    def _set_metrics(self):
        if "dstc" in self.cfg.raw_data_root.name:
            slot_categories = get_slot_categories(self.cfg.raw_data_root)
        elif "MultiWOZ" in self.cfg.raw_data_root.name:
            slot_categories = self.get_woz_slot_categories(
                self.cfg.raw_data_root.parent / "MultiWOZ_2.2"
            )
        out = []
        if self.cfg.is_multi_task:
            for task in MultiTaskNames.list():
                if task in self.cfg.multi_tasks:
                    out.append(True)
                else:
                    out.append(False)
            dst, action, response = out
        else:
            dst, action, response = [1, 1, 1]

        tod_metrics = {}
        combined_metrics = {}
        if dst:
            tod_metrics.update(
                {
                    "goal_accuracy": GoalMetric(
                        GoalMetricConfigFactory.create(GoalMetricConfigType.BELIEF),
                        slot_categories,
                    ),
                    "intent_accuracy": IntentAccuracyMetric(),
                    "requested_slots": RequestedSlotsMetric(),
                }
            )
        if action:
            tod_metrics.update(
                {
                    "inform": InformMetric(),
                    "success": SuccessMetric(slot_categories),
                    "action_accuracy": GoalMetric(
                        GoalMetricConfigFactory.create(GoalMetricConfigType.ACTION),
                        slot_categories,
                    ),
                    "user_action_accuracy": GoalMetric(
                        GoalMetricConfigFactory.create(
                            GoalMetricConfigType.USER_ACTION
                        ),
                        slot_categories,
                    ),
                }
            )
        if response:
            tod_metrics.update(
                {
                    # "response_bleu": ResponseMetric(metric_name="bleu"),
                    "response_bleu": ResponseMetric(
                        metric_name="bleu", metric_key_name="google_bleu"
                    ),
                    # "response_rouge": ResponseMetric(
                    #     # metric_name="rouge", metric_key_name="rouge2_fmeasure"
                    #     metric_name="rouge",
                    #     metric_key_name="rouge2",
                    # ),
                }
            )
        if action and response:
            combined_metrics.update(
                {
                    "combined": CombinedMetric(
                        tod_metrics["inform"],
                        tod_metrics["success"],
                        tod_metrics["response_bleu"],
                    ),
                }
            )
        self.tod_metrics = MetricCollection(tod_metrics)
        self.combined_metrics = MetricCollection(combined_metrics)

    def _remove_padding(self, text):
        return re.sub(self.cfg.padding_regexp, "", text)

    def _get_token_id(self, text: str) -> int:
        return self.cfg.tokenizer.encode(text)[0]

    def get_schemas(self, data_root: Path, step: str) -> dict[str, MultiWozSchema]:
        schemas = {}
        path = data_root / "schema.json"
        schema_json = utils.read_json(path)
        for s in schema_json:
            schema = MultiWozSchema.from_dict(s)
            schema.step = step
            schemas[schema.service_name] = schema
        return schemas

    def get_woz_slot_categories(self, data_root: Path) -> dict[str, bool]:
        schemas = self.get_schemas(data_root, Steps.TEST.value)
        out = defaultdict(bool)
        for s in schemas.values():
            for slot in s.slots:
                out[slot.name] = slot.is_categorical
        return out


def init_wandb(cfg: InferenceConfig, omega_cfg: DictConfig):
    wandb.config = omegaconf.OmegaConf.to_container(
        omega_cfg, resolve=True, throw_on_missing=True
    )
    out_dir = Path(os.getcwd())
    parent_without_year = "-".join(out_dir.parent.name.split("-")[1:])
    run_name = "/".join([parent_without_year, out_dir.name])
    group = "multi_head" if cfg.is_multi_head else "single_head"
    num_dialogs = "_".join(map(str, cfg.num_dialogs))
    tags = [cfg.model_name, num_dialogs, "inference"]
    run = wandb.init(
        name=run_name,
        group=group,
        tags=tags,
        notes=cfg.wandb.notes or "",
        project=cfg.wandb.project,
        entity="adibm",
        settings=wandb.Settings(start_method="thread"),
    )
    wandb.log({"job_id": os.environ.get("SLURM_JOB_ID", "")})


@hydra.main(config_path="../config/inference/", config_name="simple_tod_inference")
def hydra_start(cfg: DictConfig) -> None:
    with cProfile.Profile() as pr:
        inf_config = InferenceConfig(**cfg)
        utils.init_wandb(inf_config, cfg, "inference")
        inf = Inference(inf_config)
        inf.run()
        pr.dump_stats("inference.prof")


if __name__ == "__main__":
    hydra_start()
