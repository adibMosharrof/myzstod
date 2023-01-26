from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union
from omegaconf import ListConfig

import torch
from torch.utils.data import DataLoader, Dataset
from configs.dataprep_config import DataPrepConfig
from configs.dm_config import DataModuleConfig

import dstc.dstc_utils as dstc_utils
from tod.turns.zs_tod_turn import TodTurnCsvRow, TodTurnMultiHeadCsvRow
import utils
from my_enums import SpecialTokens, Steps
from simple_tod_dataclasses import (
    TodTestDataBatch,
)
from simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep
import copy


@dataclass(frozen=True)
class StepData:
    name: Steps
    num_dialog: int
    overwrite: bool
    domain_settings: Union[list[str], str]


class BaseDataModule(ABC):
    _huggingface_ignore_label_id = -100
    domain_step_map = {
        Steps.TRAIN: f"{Steps.TRAIN.value}_domain_settings",
        Steps.DEV: f"{Steps.DEV.value}_domain_settings",
        Steps.TEST: f"{Steps.TEST.value}_domain_settings",
    }

    def __init__(
        self,
        cfg: DataModuleConfig,
        steps: list[Steps],
        tod_turn_row_cls=TodTurnCsvRow,
    ):
        self.cfg = cfg
        self.tod_turn_row_cls = tod_turn_row_cls
        self.datasets: list[str, SimpleTodDataSet] = {}
        self.steps = steps
        self.setup()
        self.prompt_token_map = {}

    @abstractmethod
    def training_collator(
        self,
        batch: list[Union[TodTurnCsvRow, TodTurnMultiHeadCsvRow]],
        is_pretrain=False,
    ):
        return ValueError("Not implemented")

    @abstractmethod
    def my_test_collate(
        self, batch: list[Union[TodTurnCsvRow, TodTurnMultiHeadCsvRow]]
    ):
        return ValueError("Not implemented")

    def prepare_data(self, stdp: SimpleTODDSTCDataPrep):
        stdp.run()

    def setup_single_run(self, step, step_data, split_percent,num_dialog, domain_setting ) -> 'SimpleTodDataSet':
        cfg = copy.deepcopy(self.cfg)
        cfg.step_name = step_data.name
        cfg.num_dialogs = step_data.num_dialog
        cfg.overwrite = step_data.overwrite
        cfg.domain_setting = domain_setting
        stdp = SimpleTODDSTCDataPrep(DataPrepConfig.from_dm_config(cfg))
        self.prepare_data(stdp)
        csv_path = dstc_utils.get_csv_data_path(
            step,
            num_dialog,
            cfg=stdp.cfg,
        )
        try:
            data = utils.read_csv_dataclass(csv_path, self.tod_turn_row_cls)
            data = data[: int(len(data) * split_percent)]
        except FileNotFoundError:
            data = []
        return SimpleTodDataSet(data)

    def setup(self):
        for step, split_percent, num_dialog in zip(
            self.steps, self.cfg.data_split_percent, self.cfg.num_dialogs
        ):
            step_data = self.get_step_data(step)
            if isinstance(step_data.domain_settings[0], ListConfig):
                self.datasets[step] = []
                for domain_setting in step_data.domain_settings:
                    self.datasets[step].append(self.setup_single_run(step, step_data, split_percent, num_dialog, domain_setting))
            else:
                self.datasets[step] = self.setup_single_run(step,step_data , split_percent, num_dialog, step_data.domain_settings)
            # stdp = SimpleTODDSTCDataPrep(DataPrepConfig.from_dm_config(cfg))
            # self.prepare_data(stdp)
            # csv_path = dstc_utils.get_csv_data_path(
            #     step,
            #     num_dialog,
            #     cfg=stdp.cfg,
            # )
            # try:
            #     data = utils.read_csv_dataclass(csv_path, self.tod_turn_row_cls)
            #     data = data[: int(len(data) * split_percent)]
            # except FileNotFoundError:
            #     data = []
            # self.datasets[step] = SimpleTodDataSet(data)

    def get_step_data(self, step: Steps) -> StepData:
        index = Steps.get_index(step)
        return StepData(
            step,
            self.cfg.num_dialogs[index],
            self.cfg.overwrite[index],
            getattr(self.cfg, self.domain_step_map[step]),
        )

    def get_arguments_for_step_iteration(self, step: Steps):
        if step == Steps.TRAIN:
            return
        elif step == Steps.VALID:
            return self.val_dataloader()
        elif step == Steps.TEST:
            return self.test_dataloader()
        else:
            raise ValueError("Invalid step")

    def test_dataloader(self) -> TodTestDataBatch:
        dls = self.datasets[Steps.TEST]
        if not isinstance(dls, list):
           dls = [dls] 
        
        return [DataLoader(
            dl,
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self.my_test_collate,
            pin_memory=True,
        ) for dl in dls]

    def _get_token_id(self, text: str) -> int:
        return self.cfg.tokenizer.encode(text)[0]

    def train_tokenizer(self, item):
        try:
            tokens = self.cfg.tokenizer.encode(
                item,
                return_tensors="pt",
            )
        except TypeError as e:
            tokens = torch.empty([1, 0], dtype=torch.int64)
        return tokens.to(dtype=torch.int32)

    def contrastive_tokenizer(self, item):
        try:
            tokens = self.cfg.tokenizer(
                item,
                return_tensors="pt",
                padding="max_length",
                max_length=self.cfg.max_token_len,
            )
        except TypeError as e:
            raise ("Contrastive tokenizer failed")
        return tokens.to(dtype=torch.int32)

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

    def collate_single_item(
        self,
        context: str,
        schema: str,
        target: str,
        max_length: int,
        dont_create_labels: bool,
    ):
        context_tokens = self.train_tokenizer(context)[0]
        schema_tokens = self.train_tokenizer(schema)[0]
        target_tokens = self.train_tokenizer(target)[0]
        unused_len = (
            max_length - len(context_tokens) - len(schema_tokens) - len(target_tokens)
        )
        if len(schema_tokens) > max_length:
            raise ValueError("Schema is too long")
        if len(target_tokens) > max_length:
            raise ValueError("Target is too long")
        if unused_len < 0:
            context_start_tokens = context_tokens[:1]
            trimmed_context = context_tokens[unused_len * -1 + 1 :]
            context_tokens = torch.cat([context_start_tokens, trimmed_context], axis=0)
            unused_len = 0
        pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
        input_tokens = torch.cat([schema_tokens, context_tokens, target_tokens, pad])
        if dont_create_labels:
            label = input_tokens
        else:
            label = torch.cat(
                [
                    torch.full(
                        [len(context_tokens) + len(schema_tokens)],
                        self._huggingface_ignore_label_id,
                    ),
                    target_tokens,
                    torch.full([unused_len], self._huggingface_ignore_label_id),
                ]
            )
        attention_mask = input_tokens.ne(self.cfg.tokenizer.pad_token_id).to(
            torch.int32
        )
        # attention_mask = torch.cat(
        #     [
        #         torch.full([len(context_tokens) + len(schema_tokens) + len(target_tokens)], 1),
        #         torch.full([unused_len], 0),
        #     ]
        # )
        return input_tokens, label, attention_mask


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
