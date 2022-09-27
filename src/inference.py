import re
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2PreTrainedModel

import dstc_utils
from metrics.intent_accuracy_metric import IntentAccuracyMetric
from metrics.response_metrics import ResponseMetric
from metrics.tod_metrics_base import MetricCollection
from metrics.goal_metric import GoalMetric, GoalMetricConfigFactory
from metrics.requested_slots_metric import RequestedSlotsMetric
from metrics.dstc_metrics import InformMetric, SuccessMetric, CombinedMetric
from my_enums import DstcDomains, GoalMetricConfigType, SpecialTokens, TestSettings
import utils
from hydra_configs import DataModuleConfig, InferenceConfig
from my_datamodules import SimpleTodDataModule
from simple_tod_dataclasses import (
    InferenceRecords,
    SimpleTodConstants,
)


class Inference:
    def __init__(
        self,
        cfg: InferenceConfig,
    ):
        self.cfg = cfg

        self.tod_metrics = MetricCollection(
            {
                "goal_accuracy": GoalMetric(
                    GoalMetricConfigFactory.create(GoalMetricConfigType.BELIEF)
                ),
                "action_accuracy": GoalMetric(
                    GoalMetricConfigFactory.create(GoalMetricConfigType.ACTION)
                ),
                "intent_accuracy": IntentAccuracyMetric(),
                "requested_slots": RequestedSlotsMetric(),
                "inform": InformMetric(),
                "success": SuccessMetric(),
                "response_bleu": ResponseMetric(metric_name="bleu"),
                "response_rouge": ResponseMetric(
                    metric_name="rouge", metric_key_name="rouge2"
                ),
            }
        )
        self.bleu_metrics = MetricCollection(
            {
                "combined": CombinedMetric(
                    self.tod_metrics.metrics["inform"],
                    self.tod_metrics.metrics["success"],
                    self.tod_metrics.metrics["response_bleu"],
                ),
            }
        )

    def _get_dataloader(self):
        dm = SimpleTodDataModule(DataModuleConfig.from_inference_config(self.cfg))
        return dm.test_dataloader()

    def _get_token_id(self, token_str):
        return self.cfg.tokenizer(token_str)["input_ids"][0]

    def _remove_padding(self, text):
        return re.sub(self.cfg.padding_regexp, "", text)

    def _get_domains_from_test_settings(self, test_setting: str) -> list[str]:
        if test_setting == TestSettings.ALL:
            return DstcDomains.ALL.value
        if test_setting == TestSettings.SEEN:
            return DstcDomains.SEEN.value
        if test_setting == TestSettings.UNSEEN:
            return DstcDomains.UNSEEN.value
        if test_setting == TestSettings.CUSTOM:
            return self.cfg.domains
        raise ValueError(f"Unknown test setting {test_setting}")

    def test(self):
        self.cfg.logger.info(self.cfg.out_dir)
        for setting in self.cfg.test_settings:
            self.cfg.logger.info(f"Testing {setting}")
            domains = self._get_domains_from_test_settings(setting)
            test_csv_out_data = []
            text_csv_out_path = f"simple_tod_dstc_predictions_{setting}_{self.cfg.num_turns}_dialogs_{self.cfg.num_test_dialogs}{SimpleTodConstants.DELEXICALIZED if self.cfg.delexicalize else ''}_{'_'.join(domains)}.csv"
            test_dataloader = self._get_dataloader()
            if not len(test_dataloader):
                self.cfg.logger.info(f"No data to test for {setting}")
                continue
            inf_records = InferenceRecords()
            for batch in tqdm(test_dataloader):
                # gen = self.model.generate(
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
                    inputs=batch.context_tokens.cuda(),
                    attention_mask=batch.context_attention_masks.cuda(),
                    max_length=self.cfg.generate_max_len,
                    eos_token_id=self._get_token_id(SpecialTokens.eos_token),
                    pad_token_id=self._get_token_id(SpecialTokens.pad_token),
                    bos_token_id=self._get_token_id(SpecialTokens.bos_token),
                )
                gen_without_context = gen[:, self.cfg.max_token_len :]
                pred_text = self.cfg.tokenizer.batch_decode(
                    gen_without_context, skip_special_tokens=False
                )
                pred_text_no_pad = [self._remove_padding(text) for text in pred_text]
                if not self.cfg.is_multi_task:
                    self.tod_metrics.add_batch(
                        references=batch.targets_text, predictions=pred_text_no_pad
                    )
                    self.bleu_metrics.add_batch(
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
                self.tod_metrics.add_batch(references=refs, predictions=preds)
                self.bleu_metrics.add_batch(references=refs, predictions=preds)
            headers = ["dialog_id", "turn_id", "context", "target", "prediction"]
            utils.write_csv(headers, test_csv_out_data, text_csv_out_path)
            self.cfg.logger.info(str(self.tod_metrics))
            self.cfg.logger.info(str(self.bleu_metrics))
            self.cfg.logger.info(str(self.cfg.out_dir))
            self.tod_metrics.visualize(self.cfg.predictions_log_dir)

    def run(self):
        print("begin inference")
        self.test()
        print("end inference")
        print("-" * 80)


@hydra.main(config_path="../config/inference/", config_name="simple_tod_inference")
def hydra_start(cfg: DictConfig) -> None:
    inf = Inference(InferenceConfig(**cfg))
    inf.run()


if __name__ == "__main__":
    hydra_start()
