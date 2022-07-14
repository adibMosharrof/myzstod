from typing import Union
import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import tqdm

from transformers import GPT2LMHeadModel, GPT2PreTrainedModel, PreTrainedTokenizerFast
from dstc_dataclasses import Steps
import dstc_utils
from my_datamodules import SimpleTodDataModule, SimpleTodDataSet


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
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        dataloader: SimpleTodDataSet = None,
        tokenizer: PreTrainedTokenizerFast = None,
        model_name: str = "gpt2",
        num_test_dialogs: int = 10,
        device: str = "cuda",
        generate_max_len: int = 200,
    ):
        self.device = device
        self.project_root = Path(project_root)
        self.model = self._get_model(model)
        self.raw_data_root = Path(raw_data_root)
        self.delexicalize = delexicalize
        self.max_token_len = max_token_len
        self.data_prep_out_root = Path(data_prep_out_root)
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent
        self.model_name = model_name
        self.tokenizer = (
            tokenizer if tokenizer else dstc_utils.get_tokenizer(model_name)
        )
        self.test_num_dialogs = num_test_dialogs
        self.num_dialogs = [1, 1, self.test_num_dialogs]
        self.dataloader = dataloader if dataloader else self._get_dataloader()
        self.generate_max_len = generate_max_len

    def _get_dataloader(self):
        dm = SimpleTodDataModule(
            tokenizer=self.tokenizer,
            data_prep_out_root=self.data_prep_out_root,
            raw_data_root=self.raw_data_root,
            project_root=self.project_root,
            out_root=self.data_prep_out_root,
            eval_batch_size=self.eval_batch_size,
            data_split_percent=self.data_split_percent,
            max_token_len=self.max_token_len,
            num_workers=self.num_workers,
            delexicalize=self.delexicalize,
            num_dialogs=self.num_dialogs,
        )
        dm.setup()
        # return iter(dm.datasets[Steps.TEST])
        return dm.test_dataloader()

    def _get_model(self, model):
        if isinstance(model, str):
            model_path = self.project_root / model
            return GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        if isinstance(model, GPT2PreTrainedModel):
            return model.to(self.device)

    def test(self):

        # for context, target in self.dataloader:
        for batch in self.dataloader:
            for row in batch:
                row.context_tokens = row.context_tokens.cuda()
                row.context_attention_masks = row.context_attention_masks.cuda()
                inp = row.context_tokens[
                    torch.nonzero(row.context_attention_masks)
                ].reshape(1, -1)

                ans = self.model.generate(
                    inputs=inp,
                    do_sample=True,
                    top_k=50,
                    top_p=0.90,
                    max_length=self.generate_max_len,
                    temperature=1.5,
                )
                pred_text = self.tokenizer.decode(ans[0], skip_special_tokens=False)
                print(pred_text)
                print(row.targets_text)
                print(row.contexts_text)
                a = 1

            # ans = self.model.generate(
            #     inputs=batch.context_tokens.to(self.device),
            #     attention_mask=batch.context_attention_masks.to(self.device),
            #     do_sample=True,
            #     top_k=50,
            #     top_p=0.90,
            #     max_length=self.generate_max_len,
            #     temperature=1.5,
            # )
            # pred_text = self.tokenizer.batch_decode(ans, skip_special_tokens=False)
            # print(pred_text[0])
            a = 1

    def run(self):
        self.test()


@hydra.main(config_path="../config/inference/", config_name="simple_tod_inference")
def hydra_start(cfg: DictConfig) -> None:
    inf = Inference(
        model_name=cfg.model_name,
        project_root=cfg.project_root,
        num_workers=cfg.num_workers,
        data_split_percent=cfg.data_split_percent,
        eval_batch_size=cfg.eval_batch_size,
        max_token_len=cfg.max_token_len,
        raw_data_root=cfg.raw_data_root,
        data_prep_out_root=cfg.data_prep_out_root,
        delexicalize=cfg.delexicalize,
        model=cfg.model,
        num_test_dialogs=cfg.num_test_dialogs,
        generate_max_len=cfg.generate_max_len,
    )
    inf.run()


if __name__ == "__main__":
    hydra_start()
