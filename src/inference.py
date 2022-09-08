import re
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2PreTrainedModel

import dstc_utils
import utils
from dstc_dataclasses import DstcDomains, TestSettings
from hydra_configs import InferenceConfig
from my_datamodules import SimpleTodDataModule
from simple_tod_dataclasses import (
    GoalMetricConfigType,
    SimpleTodAction,
    SimpleTodBelief,
    SimpleTodConstants,
    SimpleTodTestDataRow,
    SpecialTokens,
    TokenizerTokens,
)
from tod_metrics import (
    CombinedMetric,
    GoalMetric,
    GoalMetricConfigFactory,
    InformMetric,
    IntentAccuracyMetric,
    MetricCollection,
    RequestedSlotsMetric,
    ResponseMetric,
    SuccessMetric,
)


class Inference:
    def __init__(
        self,
        inf_config: InferenceConfig,
    ):
        self.device = inf_config.device
        self.project_root = Path(inf_config.project_root)
        self.model = self._get_model(inf_config.model)
        self.raw_data_root = Path(inf_config.raw_data_root)
        self.delexicalize = inf_config.delexicalize
        self.max_token_len = inf_config.max_token_len
        self.data_prep_out_root = Path(inf_config.data_prep_out_root)
        self.out_dir = inf_config.out_dir
        self.predictions_log_dir = inf_config.predictions_log_dir
        self.eval_batch_size = inf_config.eval_batch_size
        self.test_batch_size = inf_config.test_batch_size
        self.num_workers = inf_config.num_workers
        self.data_split_percent = inf_config.data_split_percent
        self.model_name = inf_config.model_name
        self.tokenizer = (
            inf_config.tokenizer
            if inf_config.tokenizer
            else self._get_tokenizer(inf_config.model)
        )
        self.test_num_dialogs = inf_config.num_test_dialogs
        self.num_dialogs = [1, 1, self.test_num_dialogs]
        self.generate_max_len = inf_config.generate_max_len
        self.domains = inf_config.domains or ["restaurant", "hotel", "attraction"]
        self.num_turns = inf_config.num_turns
        self.overwrite = inf_config.overwrite
        self.padding_regexp = re.compile(re.escape(TokenizerTokens.pad_token))
        self.logger = utils.get_logger()
        self.context_max_len = inf_config.context_max_len
        self.target_max_len = inf_config.target_max_len
        self.is_multi_task = inf_config.is_multi_task
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
        self.test_settings = inf_config.test_settings

    def _get_dataloader(self, domains: list[str] = None):
        dm = SimpleTodDataModule(
            tokenizer=self.tokenizer,
            data_prep_out_root=self.data_prep_out_root,
            raw_data_root=self.raw_data_root,
            project_root=self.project_root,
            eval_batch_size=self.eval_batch_size,
            test_batch_size=self.test_batch_size,
            data_split_percent=self.data_split_percent,
            max_token_len=self.max_token_len,
            num_workers=self.num_workers,
            delexicalize=self.delexicalize,
            num_dialogs=self.num_dialogs,
            domains=domains,
            num_turns=self.num_turns,
            overwrite=self.overwrite,
            is_multi_task=self.is_multi_task,
        )
        dm.setup()
        return dm.test_dataloader()

    def _get_tokenizer(self, model_path_str):
        model_path = self.project_root / model_path_str
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path.parent.parent)
        except OSError:
            self.logger.info(
                'Could not find tokenizer for model "{}"'.format(model_path)
            )
            tokenizer = dstc_utils.get_tokenizer(self.model_name)
        return tokenizer

    def _get_model(self, model):
        if isinstance(model, str):
            model_path = self.project_root / model
            return GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        if isinstance(model, GPT2PreTrainedModel):
            return model.to(self.device)

    def _get_token_id(self, token_str):
        return self.tokenizer(token_str)["input_ids"][0]

    def _remove_padding(self, text):
        return re.sub(self.padding_regexp, "", text)

    def _get_domains_from_test_settings(self, test_setting: str) -> list[str]:
        if test_setting == TestSettings.ALL:
            return DstcDomains.ALL.value
        if test_setting == TestSettings.SEEN:
            return DstcDomains.SEEN.value
        if test_setting == TestSettings.UNSEEN:
            return DstcDomains.UNSEEN.value
        if test_setting == TestSettings.CUSTOM:
            return self.domains
        raise ValueError(f"Unknown test setting {test_setting}")

    def test(self):
        for setting in self.test_settings:
            self.logger.info(f"Testing {setting}")
            domains = self._get_domains_from_test_settings(setting)
            test_csv_out_data = []
            headers = ["target", "prediction"]
            text_csv_out_path = f"simple_tod_dstc_predictions_{setting}_{self.num_turns}_dialogs_{self.num_dialogs}{SimpleTodConstants.DELEXICALIZED if self.delexicalize else ''}_{'_'.join(domains)}.csv"
            test_dataloader = self._get_dataloader(domains)
            all_targets = []
            all_predictions = []
            for batch in tqdm(test_dataloader):
                batch: SimpleTodTestDataRow
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
                gen = self.model.generate(
                    inputs=batch.context_tokens.to(self.device),
                    attention_mask=batch.context_attention_masks.to(self.device),
                    max_length=self.generate_max_len,
                    eos_token_id=self._get_token_id(SpecialTokens.end_response),
                    pad_token_id=self._get_token_id(TokenizerTokens.pad_token),
                )
                gen_without_context = gen[:, self.max_token_len :]
                pred_text = self.tokenizer.batch_decode(
                    gen_without_context, skip_special_tokens=False
                )
                pred_text_no_pad = [self._remove_padding(text) for text in pred_text]
                self.tod_metrics.add_batch(
                    references=batch.targets_text, predictions=pred_text_no_pad
                )
                self.bleu_metrics.add_batch(
                    references=batch.targets_text, predictions=pred_text_no_pad
                )
                all_targets.append(batch.targets_text)
                all_predictions.append(pred_text_no_pad)
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            test_csv_out_data = np.column_stack([all_targets, all_predictions])
            utils.write_csv(headers, test_csv_out_data, text_csv_out_path)
            self.logger.info(str(self.tod_metrics))
            self.logger.info(str(self.bleu_metrics))
            self.logger.info(str(self.out_dir))
            self.tod_metrics.visualize(self.predictions_log_dir)

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
