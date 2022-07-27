import re
from typing import Union
import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2PreTrainedModel,
    GPT2Tokenizer,
    PreTrainedTokenizerFast,
)
import dstc_utils
from my_datamodules import SimpleTodDataModule, SimpleTodDataSet
from simple_tod_dataclasses import (
    SimpleTodAction,
    SimpleTodBelief,
    SimpleTodTestDataRow,
    SpecialTokens,
    TokenizerTokens,
)
import logging

from tod_metrics import (
    BleuMetric,
    CombinedMetric,
    GoalMetric,
    InformMetric,
    IntentAccuracyMetric,
    MetricCollection,
    RequestedSlotsMetric,
    ResponseBleuMetric,
    SuccessMetric,
)


class Inference:
    def __init__(
        self,
        model: Union[str, GPT2PreTrainedModel] = None,
        project_root: str = None,
        raw_data_root: str = None,
        delexicalize: bool = True,
        max_token_len: int = 1024,
        data_prep_out_root: str = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 32,
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        dataloader: SimpleTodDataSet = None,
        tokenizer: PreTrainedTokenizerFast = None,
        model_name: str = "gpt2",
        num_test_dialogs: int = 10,
        device: str = "cuda",
        generate_max_len: int = 1024,
        domains: list[str] = None,
        num_turns: int = 26,
        overwrite: list[bool] = None,
    ):
        self.device = device
        self.project_root = Path(project_root)
        self.model = self._get_model(model)
        self.raw_data_root = Path(raw_data_root)
        self.delexicalize = delexicalize
        self.max_token_len = max_token_len
        self.data_prep_out_root = Path(data_prep_out_root)
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer else self._get_tokenizer(model)
        self.test_num_dialogs = num_test_dialogs
        self.num_dialogs = [1, 1, self.test_num_dialogs]
        self.generate_max_len = generate_max_len
        self.domains = domains or ["restaurant", "hotel", "attraction"]
        self.num_turns = num_turns
        self.overwrite = overwrite
        self.dataloader = dataloader if dataloader else self._get_dataloader()
        self.padding_regexp = re.compile(re.escape(TokenizerTokens.pad_token))
        self.logger = logging.getLogger(__name__)
        self.tod_metrics = MetricCollection(
            {
                "goal_accuracy": GoalMetric(SimpleTodBelief),
                "action_accuracy": GoalMetric(SimpleTodAction),
                "intent_accuracy": IntentAccuracyMetric(),
                "requested_slots": RequestedSlotsMetric(),
                "inform": InformMetric(),
                "success": SuccessMetric(),
                "response_bleu": ResponseBleuMetric(),
            }
        )
        self.bleu_metrics = MetricCollection(
            {
                "overall_bleu": BleuMetric(),
                "combined": CombinedMetric(
                    self.tod_metrics.metrics["inform"],
                    self.tod_metrics.metrics["success"],
                    self.tod_metrics.metrics["response_bleu"],
                ),
            }
        )

    def _get_dataloader(self):
        dm = SimpleTodDataModule(
            tokenizer=self.tokenizer,
            data_prep_out_root=self.data_prep_out_root,
            raw_data_root=self.raw_data_root,
            project_root=self.project_root,
            out_root=self.data_prep_out_root,
            eval_batch_size=self.eval_batch_size,
            test_batch_size=self.test_batch_size,
            data_split_percent=self.data_split_percent,
            max_token_len=self.max_token_len,
            num_workers=self.num_workers,
            delexicalize=self.delexicalize,
            num_dialogs=self.num_dialogs,
            domains=self.domains,
            num_turns=self.num_turns,
            overwrite=self.overwrite,
        )
        dm.setup()
        return dm.test_dataloader()

    def _get_tokenizer(self, model_path_str):
        model_path = self.project_root / model_path_str
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path.parent)
        except OSError:
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

    def test(self):

        for batch in tqdm(self.dataloader):
            batch: SimpleTodTestDataRow
            gen = self.model.generate(
                inputs=batch.context_tokens.to(self.device),
                attention_mask=batch.context_attention_masks.to(self.device),
                do_sample=True,
                top_k=50,
                top_p=0.90,
                max_length=self.generate_max_len,
                temperature=1.5,
                eos_token_id=self._get_token_id(SpecialTokens.end_response),
                pad_token_id=self._get_token_id(TokenizerTokens.pad_token),
            )
            gen_without_context = gen[:, self.max_token_len :]
            pred_text = self.tokenizer.batch_decode(
                gen_without_context, skip_special_tokens=False
            )
            pred_text_no_pad = [self._remove_padding(text) for text in pred_text]
            self.tod_metrics.add_batch(
                references=batch.targets_text, predictions=batch.targets_text
            )
            self.bleu_metrics.add_batch(
                references=batch.targets_text, predictions=pred_text_no_pad
            )

        self.logger.info(str(self.tod_metrics))
        self.logger.info(str(self.bleu_metrics))

    def run(self):
        print("begin inference")
        self.test()
        print("end inference")
        print("-" * 80)


@hydra.main(config_path="../config/inference/", config_name="simple_tod_inference")
def hydra_start(cfg: DictConfig) -> None:
    inf = Inference(
        model_name=cfg.model_name,
        project_root=cfg.project_root,
        num_workers=cfg.num_workers,
        data_split_percent=cfg.data_split_percent,
        eval_batch_size=cfg.eval_batch_size,
        test_batch_size=cfg.test_batch_size,
        max_token_len=cfg.max_token_len,
        raw_data_root=cfg.raw_data_root,
        data_prep_out_root=cfg.data_prep_out_root,
        delexicalize=cfg.delexicalize,
        model=cfg.model,
        num_test_dialogs=cfg.num_test_dialogs,
        generate_max_len=cfg.generate_max_len,
        num_turns=cfg.num_turns,
        domains=cfg.domains,
        overwrite=cfg.overwrite,
    )
    inf.run()


if __name__ == "__main__":
    hydra_start()
