import os
import re
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2PreTrainedModel
from torch.utils.data import DataLoader
import dstc_utils
from metrics.intent_accuracy_metric import IntentAccuracyMetric
from metrics.response_metrics import ResponseMetric

# from metrics.tod_metrics_base import MetricCollection
from torchmetrics import MetricCollection
from metrics.goal_metric import GoalMetric, GoalMetricConfigFactory
from metrics.requested_slots_metric import RequestedSlotsMetric
from metrics.dstc_metrics import InformMetric, SuccessMetric, CombinedMetric
from my_enums import DstcDomains, GoalMetricConfigType, SpecialTokens, TestSettings
import utils
from hydra_configs import DataModuleConfig, InferenceConfig
from my_datamodules import TodDataModule
from simple_tod_dataclasses import (
    InferenceRecords,
    SimpleTodConstants,
    TodTestDataBatch,
)


class Inference:
    def __init__(
        self,
        cfg: InferenceConfig,
    ):
        self.cfg = cfg
        self._set_metrics()
        self.prompt_token_map = {}

    def test(self):
        self.cfg.logger.info(self.cfg.out_dir)
        for domain_setting in self.cfg.test_domain_settings:

            domains = DstcDomains[domain_setting.upper()].value
            test_csv_out_data = []
            text_csv_out_path = f"simple_tod_dstc_predictions_{domain_setting}_{self.cfg.num_turns}_dialogs_{self.cfg.num_test_dialogs}{SimpleTodConstants.DELEXICALIZED if self.cfg.delexicalize else ''}_{'_'.join(domains)}.csv"
            test_dataloader = self._get_dataloader(domain_setting)
            if not len(test_dataloader):
                self.cfg.logger.info(f"No data to test for {domain_setting}")
                continue
            inf_records = InferenceRecords()
            for batch in tqdm(test_dataloader):
                gen = self._get_generation(batch)
                gen_without_context = gen[:, self.cfg.test_prompt_max_len :]
                pred_text = self.cfg.tokenizer.batch_decode(
                    gen_without_context, skip_special_tokens=False
                )
                pred_text_no_pad = [self._remove_padding(text) for text in pred_text]
                if not self.cfg.is_multi_task:
                    self.tod_metrics.update(
                        references=batch.targets_text, predictions=pred_text_no_pad
                    )
                    self.combined_metrics.update(
                        references=batch.targets_text, predictions=pred_text_no_pad
                    )
                inf_records.add(
                    pred_text_no_pad,
                    batch.targets_text,
                    batch.dialog_ids,
                    batch.turn_ids,
                    batch.contexts_text,
                )
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
            headers = ["dialog_id", "turn_id", "context", "target", "prediction"]
            utils.write_csv(headers, test_csv_out_data, text_csv_out_path)
            self.cfg.logger.info(f"Testing {domain_setting}")
            self._print_metrics()
            self.cfg.logger.info(str(self.cfg.out_dir))
            # self.tod_metrics.visualize(self.cfg.predictions_log_dir)
            [
                self.tod_metrics[m].visualize(Path(self.cfg.predictions_log_dir))
                for m in self.tod_metrics
            ]

    def run(self):
        print("begin inference")
        self.test()
        print("end inference")
        print("-" * 80)
        print("output_dir: ", os.getcwd())

    def _get_generation(self, batch):
        # gen = self.cfg.model.generate(
        #     inputs=batch.context_tokens.to(self.device),
        #     attention_mask=batch.context_attention_masks.to(self.device),
        #     do_sample=True,
        #     top_k=50,
        #     top_p=0.94,
        #     max_length=self.generate_max_len,
        #     temperature=0.5,
        #     eos_token_id=self._get_token_id(SpecialTokens.end_response),
        #     pad_token_id=self._get_token_id(TokenizerTokens.pad_token),
        # )

        gen = self.cfg.model.generate(
            inputs=batch.input_ids.cuda(),
            attention_mask=batch.attention_masks.cuda(),
            max_length=self.cfg.generate_max_len,
            eos_token_id=self.cfg.tokenizer.eos_token_id,
            pad_token_id=self.cfg.tokenizer.pad_token_id,
            bos_token_id=self.cfg.tokenizer.bos_token_id,
        )
        return gen

    def _print_metrics(self):
        tod_metrics_str = [str(self.tod_metrics[m]) for m in self.tod_metrics]
        combined_metrics_str = [
            str(self.combined_metrics[m]) for m in self.combined_metrics
        ]
        all_metric_str = "\n".join(
            np.concatenate([tod_metrics_str, combined_metrics_str])
        )
        # tod_metric_
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

    def _set_metrics(self):
        dst, action, response = (
            self.cfg.multi_tasks if self.cfg.is_multi_task else [1, 1, 1]
        )
        tod_metrics = {}
        combined_metrics = {}
        if dst:
            tod_metrics.update(
                {
                    "goal_accuracy": GoalMetric(
                        GoalMetricConfigFactory.create(GoalMetricConfigType.BELIEF)
                    ),
                    "action_accuracy": GoalMetric(
                        GoalMetricConfigFactory.create(GoalMetricConfigType.ACTION)
                    ),
                    "intent_accuracy": IntentAccuracyMetric(),
                    "requested_slots": RequestedSlotsMetric(),
                }
            )
        if action:
            tod_metrics.update(
                {
                    "inform": InformMetric(),
                    "success": SuccessMetric(),
                }
            )
        if response:
            tod_metrics.update(
                {
                    "response_bleu": ResponseMetric(metric_name="bleu"),
                    "response_rouge": ResponseMetric(
                        metric_name="rouge", metric_key_name="rouge2"
                    ),
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

    def _get_dataloader(self, test_setting: str) -> TodTestDataBatch:
        dm = TodDataModule(
            DataModuleConfig.from_inference_config(
                self.cfg,
                domain_setting=test_setting,
            )
        )
        return dm.test_dataloader()

    def _remove_padding(self, text):
        return re.sub(self.cfg.padding_regexp, "", text)

    def _get_token_id(self, text: str) -> int:
        return self.cfg.tokenizer.encode(text)[0]


@hydra.main(config_path="../config/inference/", config_name="simple_tod_inference")
def hydra_start(cfg: DictConfig) -> None:
    inf = Inference(InferenceConfig(**cfg))
    inf.run()


if __name__ == "__main__":
    hydra_start()
