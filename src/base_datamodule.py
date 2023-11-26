from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union
from omegaconf import ListConfig

import torch
from torch.utils.data import DataLoader, Dataset
from configs.dataprep_config import DataPrepConfig
from configs.dm_config import DataModuleConfig
from configs.multi_woz_data_prep_config import MultiWozDataPrepConfig
from data_prep.data_prep_strategy_resolver import DataPrepStrategyResolver
from data_prep.dstc_base_data_prep import DstcBaseDataPrep

from multi_woz.tod_multi_woz_21_data_prep import TodMultiWoz21DataPrep
from multi_woz.tod_multi_woz_22_data_prep import TodMultiWoz22DataPrep
from tod.turns.zs_tod_turn import TodTurnCsvRow, TodTurnMultiHeadCsvRow
import utils
from my_enums import SpecialTokens, Steps, MultiTaskNames
from simple_tod_dataclasses import (
    TodTestDataBatch,
)
from data_prep.simple_tod_dstc_data_prep import SimpleTODDSTCDataPrep
import copy
import pandas as pd
import random

random.seed(420)


@dataclass(frozen=True)
class StepData:
    name: Steps
    num_dialog: int
    overwrite: bool
    split_percent: float
    domain_settings: Union[list[str], str]


@dataclass(frozen=False)
class TodTrainRowCollator:
    input_tokens: torch.IntTensor
    label: torch.IntTensor
    attention_mask: torch.IntTensor


@dataclass(frozen=False)
class ScaleGradRowCollator(TodTrainRowCollator):
    mt_prompt_token_ids: Optional[torch.IntTensor] = None
    special_tokens_target_mask: Optional[torch.IntTensor] = None
    special_tokens_vocab_mask: Optional[torch.IntTensor] = None


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
        task_name: Optional[MultiTaskNames] = None,
    ):
        self.cfg = cfg
        self.task_name = task_name
        self.tod_turn_row_cls = tod_turn_row_cls
        self.datasets: dict[str, SimpleTodDataSet] = {}
        self.grouped_test_datasets: list[str, SimpleTodDataSet] = {}
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
        with self.cfg.accelerator.main_process_first():
            stdp.run()

    def get_data_prep_class(self, cfg: DataModuleConfig):
        if isinstance(cfg.raw_data_root, str):
            cfg.raw_data_root = Path(cfg.raw_data_root)
        try:
            dp_cfg = DataPrepConfig.from_dm_config(cfg)
            return DstcBaseDataPrep(dp_cfg, DataPrepStrategyResolver.resolve(dp_cfg))
        except ValueError:
            pass
        if "MultiWOZ_2.2" in cfg.raw_data_root.name:
            return TodMultiWoz22DataPrep(MultiWozDataPrepConfig.from_dm_config(cfg))
        if "MultiWOZ_2.1" in cfg.raw_data_root.name:
            return TodMultiWoz21DataPrep(MultiWozDataPrepConfig.from_dm_config(cfg))
        elif "dstc" in cfg.raw_data_root.name:
            return SimpleTODDSTCDataPrep(DataPrepConfig.from_dm_config(cfg))

    def setup_single_run(
        self, step: str, step_data: StepData, domain_setting: Union[str, list[str]]
    ) -> "SimpleTodDataSet":
        cfg = copy.deepcopy(self.cfg)
        cfg.step_name = step_data.name
        cfg.num_dialogs = step_data.num_dialog
        cfg.overwrite = step_data.overwrite
        cfg.domain_setting = domain_setting

        data_prep = self.get_data_prep_class(cfg)
        self.prepare_data(data_prep)
        csv_path = utils.get_csv_data_path(
            step,
            step_data.num_dialog,
            cfg=data_prep.cfg,
        )
        try:
            data = utils.read_csv_dataclass(csv_path, self.tod_turn_row_cls)
        except FileNotFoundError:
            data = []

        if self.cfg.is_multi_task:
            if step == Steps.TEST.value:
                data = self.combine_tasks_for_inference(data)
            if self.task_name:
                data = self.filter_data_by_task_name(data, self.task_name)

        data = self.get_data_by_split_percent(data, step_data.split_percent)
        return SimpleTodDataSet(data)

    def combine_tasks_for_inference(self, data: list[TodTurnCsvRow]):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        if data.empty:
            return data
        grouped = (
            data.groupby(["dialog_id", "turn_id"])
            .agg({"context": "last", "schema": "last", "target": "sum"})
            .reset_index()
        )
        # grouped_wo_dup = grouped.drop_duplicates()
        # return grouped
        return [
            self.tod_turn_row_cls(**row) for row in grouped.to_dict(orient="records")
        ]

    def filter_data_by_task_name(self, data: list, task_name: MultiTaskNames):
        return [d for d in data if d.task == task_name.value]

    def setup(self):
        for step in self.steps:
            step_data = self.get_step_data(step)
            if isinstance(step_data.domain_settings[0], (list, ListConfig)):
                self.datasets[step] = []
                for domain_setting in step_data.domain_settings:
                    self.datasets[step].append(
                        self.setup_single_run(step, step_data, domain_setting)
                    )
            else:
                self.datasets[step] = self.setup_single_run(
                    step,
                    step_data,
                    step_data.domain_settings,
                )
        if self.cfg.test_num_turns_groups:
            self.create_test_data_grouped_by_dialog_turns(self.datasets[Steps.TEST])
        if self.cfg.create_data_from_train:
            self.create_dev_test_from_train()
            if self.cfg.is_multi_task:
                for d in self.datasets[Steps.TEST]:
                    d.data = self.combine_tasks_for_inference(d.data)
                a = 1

    """
        Test domain must be an array of length 1.
        That one test array can have multiple domains in it.
        No support for multiple different independent test domains
    """

    def create_dev_test_from_train(self):
        if Steps.TRAIN in self.steps:
            if not self.datasets[Steps.TRAIN]:
                self._check_if_step_data_exists(Steps.TRAIN)
        if Steps.TRAIN.value not in self.datasets:
            # TODO: Try to remove the hardcoded indexes
            train_step_data = StepData(
                name=Steps.TRAIN.value,
                num_dialog=self.cfg.num_dialogs[0],
                overwrite=self.cfg.overwrite[0],
                split_percent=self.cfg.data_split_percent[0],
                domain_settings=self.cfg.test_domain_settings[0],
            )
            self.datasets[Steps.TRAIN.value] = self.setup_single_run(
                Steps.TRAIN.value,
                train_step_data,
                train_step_data.domain_settings,
            )
            self._check_if_step_data_exists(
                Steps.TRAIN,
                "There is no train data, so cannot create dev/test, TIP: Try passing more dialog files",
            )

        for step, split_percent in zip(
            [Steps.DEV.value, Steps.TEST.value], self.cfg.create_data_from_train_splits
        ):
            if step == Steps.DEV:
                ds = getattr(self.datasets, step, None)
            elif step == Steps.TEST:
                test_ds = getattr(self.datasets, step, None)
                ds = test_ds[0] if test_ds else None
            if ds and ds.data:
                continue
            train_df = pd.DataFrame(self.datasets[Steps.TRAIN].data)
            train_dialog_ids = list(train_df.dialog_id.unique())
            if len(train_dialog_ids) < 3:
                raise ValueError(
                    "There are not enough train dialogs to create dev/test"
                )
            new_data_dialog_ids = random.sample(
                train_dialog_ids, math.ceil(len(train_dialog_ids) * split_percent)
            )
            new_data = train_df[train_df.dialog_id.isin(new_data_dialog_ids)]
            if self.cfg.is_multi_task and step == Steps.TEST:
                new_data = self.combine_tasks_for_inference(new_data)
            updated_train_data = train_df[~train_df.dialog_id.isin(new_data_dialog_ids)]
            self.datasets[Steps.TRAIN] = SimpleTodDataSet(
                [
                    self.tod_turn_row_cls(**row)
                    for row in updated_train_data.to_dict(orient="records")
                ]
            )
            if isinstance(new_data, list):
                new_step_data = SimpleTodDataSet(new_data)
            elif isinstance(new_data, pd.DataFrame):
                curr_data = new_data.to_dict(orient="records")
                new_step_data = SimpleTodDataSet(
                    [self.tod_turn_row_cls(**row) for row in curr_data]
                )
            self.datasets[step] = (
                [new_step_data] if step == Steps.TEST else new_step_data
            )

    def create_test_data_grouped_by_dialog_turns(
        self, datasets: list[list[TodTurnCsvRow]]
    ):
        out = {}
        for domain in self.cfg.test_domain_settings:
            for group in self.cfg.test_num_turns_groups:
                domain_str = "_".join(domain)
                key = f"{domain_str}_{group}"
                out[key] = []

        for dataset, domain in zip(datasets, self.cfg.test_domain_settings):
            bins = [0] + self.cfg.test_num_turns_groups
            labels = self.cfg.test_num_turns_groups
            df = pd.DataFrame(dataset.data)
            dialog_turn_counts = (
                df.groupby("dialog_id").size().reset_index(name="counts")
            )
            dialog_turn_counts["binned"] = pd.cut(
                dialog_turn_counts["counts"], bins=bins, labels=labels
            )
            domain_str = "_".join(domain)
            for dialog_id, turn_count, bin_id in dialog_turn_counts.values:
                mask = df["dialog_id"] == dialog_id
                key = f"{domain_str}_{bin_id}"
                out[key].append(df[mask])
        for key, data in out.items():
            try:
                ds_data = pd.concat(data, axis=0)
                row_data = ds_data.apply(
                    lambda row: self.tod_turn_row_cls(*row), axis=1
                ).tolist()
            except ValueError:
                row_data = []
            self.grouped_test_datasets[key] = SimpleTodDataSet(row_data)

    def get_step_data(self, step: Steps) -> StepData:
        index = Steps.get_index(step)
        return StepData(
            step,
            self.cfg.num_dialogs[index],
            self.cfg.overwrite[index],
            self.cfg.data_split_percent[index],
            getattr(self.cfg, self.domain_step_map[step]),
        )

    def get_data_by_split_percent(
        self, data: list[TodTurnCsvRow], split_percent: float
    ):
        return data[: int(len(data) * split_percent)]

    def _check_if_step_data_exists(
        self,
        step: Steps = Steps.TRAIN,
        msg: str = "There is no train data, so cannot create dev/test",
    ):
        if not self.datasets[step]:
            raise ValueError(msg)

    def test_dataloader(self) -> list[Tuple[TodTestDataBatch, str]]:
        dls = self.datasets[Steps.TEST]
        if not isinstance(dls, list):
            dls = [dls]

        return [
            (
                DataLoader(
                    dl,
                    batch_size=self.cfg.test_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    collate_fn=self.my_test_collate,
                    pin_memory=True,
                ),
                domain_setting,
            )
            for dl, domain_setting in zip(dls, self.cfg.test_domain_settings)
        ]

    def grouped_test_dataloader(self) -> list[Tuple[TodTestDataBatch, str]]:
        return [
            (
                DataLoader(
                    dl,
                    batch_size=self.cfg.test_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_workers,
                    collate_fn=self.my_test_collate,
                    pin_memory=True,
                ),
                key,
            )
            for key, dl in self.grouped_test_datasets.items()
        ]

    def _get_token_id(self, text: str) -> int:
        return self.cfg.tokenizer.encode(text)[0]

    def train_tokenizer(self, item):
        try:
            tokens = self.cfg.tokenizer.encode(
                item,
                return_tensors="pt",
            )
        except Exception as e:
            tokens = torch.empty([1, 0], dtype=torch.int32)
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

    def t5_collate_single_item(
        self, item: TodTurnCsvRow, max_length: int
    ) -> TodTrainRowCollator:
        context_tokens = self.train_tokenizer(item.context)[0]
        schema_tokens = self.train_tokenizer(item.schema)[0]
        target_tokens = self.train_tokenizer(item.target)[0]

        prompt_text = "\n".join(
            [
                "Instructions: Given the Dialog History and the Dialog Schemas, please generate the system response.\n",
                "Dialog History\n",
            ]
        )
        prompt_tokens = self.train_tokenizer(prompt_text)[0]
        schema_prompt_text = "\n\nDialog Schemas\n"
        schema_prompt_tokens = self.train_tokenizer(schema_prompt_text)[0]
        context_unused_len = (
            self.cfg.test_prompt_max_len
            - len(prompt_tokens)
            - len(context_tokens)
            - len(schema_prompt_tokens)
            - len(schema_tokens)
        )
        if len(schema_tokens) > self.cfg.test_prompt_max_len:
            raise ValueError("Schema is too long")
        target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
        if len(target_tokens) > target_max_len:
            raise ValueError("Target is too long")

        if context_unused_len < 0:
            context_tokens = context_tokens[context_unused_len * -1 :]
            context_unused_len = 0
        pad = torch.full([context_unused_len], self.cfg.tokenizer.pad_token_id)
        input_tokens = torch.cat(
            [prompt_tokens, context_tokens, schema_prompt_tokens, schema_tokens, pad]
        )

        target_unused_len = target_max_len - len(target_tokens) - 1
        label = torch.cat(
            [
                target_tokens,
                torch.full([target_unused_len], self._huggingface_ignore_label_id),
                torch.full([1], self.cfg.tokenizer.eos_token_id),
            ]
        )

        attention_mask = input_tokens.ne(self.cfg.tokenizer.pad_token_id).to(
            torch.int32
        )

        return TodTrainRowCollator(input_tokens, label, attention_mask)

    def collate_single_item(
        self,
        item: TodTurnCsvRow,
        max_length: int,
        dont_create_labels: bool,
        is_t5_model: bool = False,
    ) -> TodTrainRowCollator:
        context_tokens = self.train_tokenizer(item.context)[0]
        schema_tokens = self.train_tokenizer(item.schema)[0]
        target_tokens = self.train_tokenizer(item.target)[0]
        prompt_tokens = torch.tensor([], dtype=torch.torch.int32)
        if is_t5_model:
            prompt_text = "\n".join(
                [
                    "Instructions: Given the dialog history and the schemas, please generate the system response.\n\n",
                    "Begin Context",
                    "Dialog History",
                ]
            )
            prompt_tokens = self.train_tokenizer(prompt_text)[0]

        unused_len = (
            max_length
            - len(context_tokens)
            - len(schema_tokens)
            - len(target_tokens)
            - len(prompt_tokens)
        )
        if len(schema_tokens) > max_length:
            raise ValueError("Schema is too long")
        if not dont_create_labels and len(target_tokens) > max_length:
            raise ValueError("Target is too long")
        if unused_len < 0:
            # raise ValueError("Need larger token length")
            context_start_tokens = context_tokens[:1]
            trimmed_context = context_tokens[unused_len * -1 + 1 :]
            context_tokens = torch.cat(
                [prompt_tokens, context_start_tokens, trimmed_context], axis=0
            )
            unused_len = 0
        pad = torch.full([unused_len], self.cfg.tokenizer.pad_token_id)
        input_tokens = torch.cat([schema_tokens, context_tokens, target_tokens, pad])
        if is_t5_model:
            label = torch.cat(
                [
                    input_tokens,
                    torch.full([unused_len], self._huggingface_ignore_label_id),
                ]
            )

        elif dont_create_labels:
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
        special_tokens_mask = None
        if not self.cfg.is_scale_grad:
            return TodTrainRowCollator(input_tokens, label, attention_mask)

        special_tokens_mask, special_tokens_vocab_mask = self.get_special_tokens(
            input_tokens,
            item.special_tokens,
            context_tokens,
            schema_tokens,
            target_tokens,
            unused_len,
            max_length,
        )
        return ScaleGradRowCollator(
            input_tokens,
            label,
            attention_mask,
            special_tokens_target_mask=special_tokens_mask,
            special_tokens_vocab_mask=special_tokens_vocab_mask,
        )
        return input_tokens, label, attention_mask

    def get_special_tokens(
        self,
        input_tokens,
        special_tokens,
        context_tokens,
        schema_tokens,
        target_tokens,
        unused_len,
        max_length,
    ):
        special_token_ids = self.train_tokenizer(special_tokens)
        ignore_ids = torch.tensor([3, 140, 142])
        ignore_ids = torch.tensor([2, 139])
        filtered_ids = torch.masked_select(
            special_token_ids, ~torch.isin(special_token_ids, ignore_ids)
        )
        context_schema_len = len(context_tokens) + len(schema_tokens)
        mask = torch.isin(
            input_tokens[context_schema_len : context_schema_len + len(target_tokens)],
            filtered_ids,
        )
        special_tokens_target_mask = torch.cat(
            [
                torch.full([context_schema_len], 0),
                mask,
                torch.full([unused_len], 0),
            ]
        )
        special_tokens_vocab_mask = torch.zeros(
            max_length, len(self.cfg.tokenizer), dtype=torch.bool
        )
        special_tokens_vocab_mask.scatter_(1, special_token_ids.long(), 1)
        return special_tokens_target_mask, special_tokens_vocab_mask


class SimpleTodDataSet(Dataset):
    def __init__(
        self,
        data: List[TodTurnCsvRow],
    ):
        self.data: list[TodTurnCsvRow] = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> TodTurnCsvRow:
        return self.data[idx]
