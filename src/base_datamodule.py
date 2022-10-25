from pathlib import Path
from typing import Dict, Iterable, List

import torch
from responses import target
from torch.utils.data import DataLoader, Dataset

import dstc_utils
import utils
from hydra_configs import DataModuleConfig, DataPrepConfig
from my_enums import SpecialTokens, Steps
from simple_tod_dataclasses import TodTestDataBatch, TodTurnCsvRow
from simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep


class BaseDataModule:
    steps = Steps.list()
    _huggingface_ignore_label_id = -100

    def __init__(
        self,
        cfg: DataModuleConfig,
    ):
        self.cfg = cfg
        self.setup()
        self.prompt_token_map = {}

    def _get_token_id(self, text: str) -> int:
        return self.cfg.tokenizer.encode(text)[0]

    def prepare_data(self):
        stdp = SimpleTODDSTCDataPrep(DataPrepConfig.from_dm_config(self.cfg))
        stdp.run()

    def setup(self):
        self.prepare_data()
        for step, split_percent, num_dialog in zip(
            self.steps, self.cfg.data_split_percent, self.cfg.num_dialogs
        ):
            csv_path = dstc_utils.get_csv_data_path(
                step,
                num_dialog,
                cfg=self.cfg,
            )
            try:
                data = utils.read_csv_dataclass(csv_path, TodTurnCsvRow)
                data = data[: int(len(data) * split_percent)]
            except FileNotFoundError:
                data = []
            self.cfg.datasets[step] = SimpleTodDataSet(data)

    def test_dataloader(self) -> TodTestDataBatch:
        return DataLoader(
            self.cfg.datasets[Steps.TEST],
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.my_test_collate,
            pin_memory=True,
        )

    def train_tokenizer(self, item):
        try:
            tokens = self.cfg.tokenizer.encode(
                item,
                return_tensors="pt",
            )
        except TypeError as e:
            tokens = torch.empty([1, 0], dtype=torch.int64)
        return tokens

    def contrastive_tokenizer(self, item):
        try:
            tokens = self.cfg.tokenizer.encode(
                item,
                return_tensors="pt",
                padding="max_length",
                max_length=self.cfg.contrastive_max_token_len,
            )
        except TypeError as e:
            tokens = torch.empty([1, 0], dtype=torch.int64)
        return tokens[0]

    def get_training_labels(self, context_len, unused_len, target_tokens):
        return torch.cat(
            [
                torch.full([context_len], self._huggingface_ignore_label_id),
                target_tokens,
                torch.full([unused_len], self._huggingface_ignore_label_id),
            ]
        )

    def pretraining_collator(self, batch: list[TodTurnCsvRow]):
        return self.training_collator(batch, True)

    def _get_filler_token_from_prompt(self, prompt_token: int, prompt_token_map: dict):
        try:
            filler_token = prompt_token_map[int(prompt_token)]
        except KeyError:
            raise ValueError("Prompt token not found")
        return filler_token

    def _add_multi_task_prompt_token(
        self, context_tokens: torch.Tensor
    ) -> torch.Tensor:
        if not self.prompt_token_map.keys():
            self.prompt_token_map = {
                self._get_token_id(SpecialTokens.prompt_dst): self._get_token_id(
                    SpecialTokens.begin_dsts
                ),
                self._get_token_id(SpecialTokens.prompt_action): self._get_token_id(
                    SpecialTokens.begin_action
                ),
                self._get_token_id(SpecialTokens.prompt_response): self._get_token_id(
                    SpecialTokens.begin_response
                ),
            }
        prompt_token = self._get_filler_token_from_prompt(
            context_tokens[-2], self.prompt_token_map
        )
        out = torch.cat([context_tokens, torch.tensor([prompt_token])])
        return out


class SimpleTodDataSet(Dataset):
    def __init__(
        self,
        data: List[TodTurnCsvRow],
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> TodTurnCsvRow:
        return self.data[idx]
