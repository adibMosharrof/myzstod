import os
import random
import re
from pathlib import Path
from typing import Dict, Union

from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2PreTrainedModel
from dstc_dataclasses import DstcSchema

import dstc_utils
import utils
from my_enums import (
    ContextType,
    ContrastiveConstants,
    DstcDomains,
    SpecialTokens,
    Steps,
)
from multi_head.mh_model import GPT2MultiLMHeadModel


class TrainerConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        data_prep_out_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        model_name: str = "gpt2",
        contrastive_model_name: str = "sentence-transformers/stsb-roberta-base-v2",
        tokenizer_name: str = "gpt2",
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        early_stopping_patience: int = 3,
        eval_steps: int = 500,
        eval_batch_size: int = 6,
        test_batch_size: int = 32,
        train_batch_size: int = 8,
        pretrain_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        n_layer: int = 12,
        n_head: int = 12,
        contrastive_train_batch_size: int = 100,
        num_dialogs: list[int] = None,
        delexicalize: bool = False,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_setting: str = None,
        train_domain_percentage: float = 1.0,
        test_domain_settings: list[str] = None,
        out_dir: str = "results",
        pretrain_epochs: int = 1,
        pretrain_model_path: str = None,
        train_model_path: str = None,
        train_epochs: int = 1,
        contrastive_train_epochs: int = 3,
        logging_dir: str = "logs",
        generate_max_len: int = 1024,
        should_test: bool = False,
        logging_steps: int = 50,
        test_prompt_max_len: int = 799,
        max_token_len: int = 1022,
        eval_accumulation_steps: int = 16,
        is_multi_head: bool = False,
        is_multi_task: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_user_actions: bool = False,
        should_add_sys_actions: bool = False,
        ce_loss_weight: float = 0.50,
        contrastive_loss_weight: float = 0.50,
        contrastive_model: str = None,
        contrast_with: list[str] = None,
        contrastive_max_token_len: int = 250,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        should_add_dsts: bool = False,
        single_action_neg_samples: int = 10,
        local_rank: int = 0,
    ) -> None:
        self.project_root = Path(project_root)
        self.data_prep_out_root = Path(data_prep_out_root)
        self.model_name = model_name
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.early_stopping_patience = early_stopping_patience
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_token_len = max_token_len
        self.num_dialogs = num_dialogs or [20, 10, 17]
        self.delexicalize = delexicalize
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.test_domain_settings = test_domain_settings or ["all", "seen", "unseen"]
        self.out_dir = Path(out_dir)
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.contrastive_train_epochs = contrastive_train_epochs
        self.train_domain_setting = train_domain_setting
        self.train_domain_percentage = train_domain_percentage
        self.pretrain_model_path = pretrain_model_path
        self.train_model_path = train_model_path
        self.logging_dir = Path(logging_dir)
        self.generate_max_len = generate_max_len
        self.should_test = should_test
        self.delexicalize = delexicalize
        self.logging_steps = logging_steps
        self.train_batch_size = train_batch_size
        self.contrastive_train_batch_size = contrastive_train_batch_size
        self.pretrain_batch_size = pretrain_batch_size
        self.raw_data_root = self.project_root / raw_data_root
        self.test_prompt_max_len = test_prompt_max_len
        self.eval_accumulation_steps = eval_accumulation_steps
        self.is_multi_head = is_multi_head
        self.is_multi_task = is_multi_task
        self.multi_tasks = (
            multi_tasks if self.is_multi_task and multi_tasks else [1, 1, 1]
        )
        # self.tokenizer = dstc_utils.get_tokenizer(model_name)
        self.tokenizer = dstc_utils.get_tokenizer(tokenizer_name)
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        if test_prompt_max_len > max_token_len:
            raise ValueError("context_max_len must be less than max_token_len")
        self.ce_loss_weight = ce_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_model = contrastive_model
        self.contrast_with = contrast_with or []
        self.contrastive_max_token_len = contrastive_max_token_len
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results
        self.tokenizer_name = tokenizer_name
        self.contrastive_model_name = contrastive_model_name
        self.should_add_dsts = should_add_dsts
        self.single_action_neg_samples = single_action_neg_samples
        self.local_rank = local_rank


class InferenceConfig:
    def __init__(
        self,
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 100,
        max_token_len: int = 512,
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        data_prep_out_root: str = "processed_data/simple_tod",
        predictions_log_dir: str = "predictions_logs",
        num_test_dialogs: int = 17,
        delexicalize: bool = False,
        model: str = "outputs/2022-07-26/22-28-09/results/train/checkpoint-7067",
        model_name: str = "gpt2",
        generate_max_len: int = 1024,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_percentage: float = 1.0,
        test_domain_settings: list[str] = None,
        out_dir: str = "results",
        tokenizer: AutoTokenizer = None,
        test_prompt_max_len: int = 799,
        is_multi_task: bool = False,
        is_multi_head: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_user_actions: bool = False,
        should_add_sys_actions: bool = False,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.max_token_len = max_token_len
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.data_prep_out_root = data_prep_out_root
        self.num_test_dialogs = num_test_dialogs
        self.delexicalize = delexicalize
        self.is_multi_head = is_multi_head
        self.model = self._get_model(model)
        self.model_name = model_name
        self.generate_max_len = generate_max_len
        self.train_domain_percentage = train_domain_percentage
        self.test_domain_settings = test_domain_settings or ["all", "seen", "unseen"]
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.out_dir = out_dir
        self.test_prompt_max_len = test_prompt_max_len
        self.predictions_log_dir = Path(predictions_log_dir)
        self.predictions_log_dir.mkdir(parents=True, exist_ok=True)
        self.is_multi_task = is_multi_task
        self.multi_tasks = (
            multi_tasks if self.is_multi_task and multi_tasks else [1, 1, 1]
        )
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.logger = utils.get_logger()
        self.tokenizer = tokenizer if tokenizer else self._get_tokenizer(model)
        self.padding_regexp = re.compile(re.escape(SpecialTokens.pad_token))
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results
        # self.contrastive_model = contrastive_model

    def _get_tokenizer(self, model_path_str: str):
        model_path: Path = self.project_root / model_path_str
        try:
            # with specifig checkpoint number (results/train/checkpoint-1000)
            tok_path = model_path.parent.parent.parent / "tokenizer"
            # checkpoint not provided (results/train)
            if not tok_path.exists():
                tok_path = model_path.parent.parent / "tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except OSError:
            self.logger.info(
                'Could not find tokenizer for model "{}"'.format(model_path)
            )
            tokenizer = dstc_utils.get_tokenizer(self.model_name)
        return tokenizer

    def _get_model(self, model):
        if isinstance(model, str):
            model_path = self.project_root / model
            if self.is_multi_head:
                return GPT2MultiLMHeadModel.from_pretrained(model_path).cuda()
            return GPT2LMHeadModel.from_pretrained(model_path).cuda()
        if isinstance(model, GPT2PreTrainedModel):
            return model.cuda()

    @classmethod
    def from_trainer_config(
        cls, trainer_config: TrainerConfig, model: GPT2LMHeadModel
    ) -> "InferenceConfig":
        return cls(
            num_workers=trainer_config.num_workers,
            data_split_percent=trainer_config.data_split_percent,
            eval_batch_size=trainer_config.eval_batch_size,
            test_batch_size=trainer_config.test_batch_size,
            max_token_len=trainer_config.max_token_len,
            raw_data_root=trainer_config.raw_data_root,
            project_root=trainer_config.project_root,
            data_prep_out_root=trainer_config.data_prep_out_root,
            num_test_dialogs=trainer_config.num_dialogs[2],
            delexicalize=trainer_config.delexicalize,
            model=model,
            model_name=trainer_config.model_name,
            generate_max_len=trainer_config.generate_max_len,
            test_domain_settings=trainer_config.test_domain_settings,
            num_turns=trainer_config.num_turns,
            overwrite=trainer_config.overwrite,
            out_dir=trainer_config.out_dir,
            tokenizer=trainer_config.tokenizer,
            test_prompt_max_len=trainer_config.test_prompt_max_len,
            is_multi_task=trainer_config.is_multi_task,
            multi_tasks=trainer_config.multi_tasks,
            should_add_schema=trainer_config.should_add_schema,
            should_add_sys_actions=trainer_config.should_add_sys_actions,
            should_add_user_actions=trainer_config.should_add_user_actions,
            context_type=trainer_config.context_type,
            should_add_service_results=trainer_config.should_add_service_results,
            # contrastive_model=trainer_config.contrastive_model,
        )


class DataModelExplorationConfig:
    def __init__(
        self,
        data_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        project_root: str = None,
        num_dialogs: list[int] = None,
        delexicalize: bool = False,
        model_name: str = "gpt2",
        out_root: str = "model_exploration",
        num_turns: int = 10,
        domain_setting: str = "SEEN",
        overwrite: list[bool] = None,
        data_split_percent: list[float] = None,
        is_multi_task: bool = False,
        should_add_schema: bool = False,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.raw_data_root = self.project_root / raw_data_root
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.tokenizer = dstc_utils.get_tokenizer(model_name)
        self.num_turns = num_turns
        self.is_multi_task = is_multi_task
        self.should_add_schema = should_add_schema
        self.overwrite = overwrite or [False, False, False]
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.domain_setting = domain_setting
        self.domains = DstcDomains[domain_setting.upper()].value


class ContrastiveConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        data_prep_out_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        contrastive_model_name: str = "sentence-transformers/stsb-distilroberta-base-v2",
        tokenizer_name: str = "gpt2",
        model: str = None,
        data_split_percent: list[float] = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 32,
        contrastive_train_batch_size: int = 8,
        num_dialogs: list[int] = None,
        num_turns: int = 10,
        num_workers: int = 8,
        overwrite: list[bool] = None,
        train_domain_setting: str = "ALL",
        test_domain_settings: list[str] = None,
        out_dir: str = "results",
        train_epochs: int = 2,
        logging_dir: str = "logs",
        logging_steps: int = 50,
        eval_accumulation_steps: int = 5,
        is_multi_task: bool = False,
        multi_tasks: list[int] = None,
        contrast_with: list[str] = None,
        single_action_neg_samples: int = 10,
        should_add_dsts: bool = False,
        contrastive_max_token_len: int = 512,
    ):
        self.project_root = Path(project_root)
        self.data_prep_out_root = self.project_root / data_prep_out_root
        self.raw_data_root = Path(raw_data_root)
        self.contrastive_model_name = contrastive_model_name
        self.model = self.project_root / model if model else None
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.contrastive_train_batch_size = contrastive_train_batch_size
        self.num_dialogs = num_dialogs or [20, 5, 10]
        self.num_turns = num_turns
        self.num_workers = num_workers
        self.overwrite = overwrite or [False, False, False]
        self.train_domain_setting = train_domain_setting
        self.test_domain_settings = test_domain_settings or ["ALL", "SEEN", "UNSEEN"]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.train_epochs = train_epochs
        self.logging_dir = Path(logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.logging_steps = logging_steps
        self.eval_accumulation_steps = eval_accumulation_steps
        self.is_multi_task = is_multi_task
        self.multi_tasks = multi_tasks or [1, 1, 1]
        self.contrast_with = (
            contrast_with if contrast_with else [ContrastiveConstants.USER_ACT]
        )
        self.single_action_neg_samples = single_action_neg_samples
        self.should_add_user_actions = (
            True if ContrastiveConstants.USER_ACT in self.contrast_with else False
        )
        self.should_add_dsts = should_add_dsts
        self.tokenizer_name = tokenizer_name
        self.contrastive_max_token_len = contrastive_max_token_len

    @classmethod
    def from_trainer_config(self, trainer_cfg: TrainerConfig) -> "ContrastiveConfig":
        return self(
            project_root=trainer_cfg.project_root,
            data_prep_out_root=trainer_cfg.data_prep_out_root,
            raw_data_root=trainer_cfg.raw_data_root,
            out_dir=trainer_cfg.out_dir,
            contrastive_model_name=trainer_cfg.contrastive_model_name,
            tokenizer_name=trainer_cfg.tokenizer_name,
            num_dialogs=trainer_cfg.num_dialogs,
            num_turns=trainer_cfg.num_turns,
            num_workers=trainer_cfg.num_workers,
            overwrite=trainer_cfg.overwrite,
            train_domain_setting=trainer_cfg.train_domain_setting,
            is_multi_task=trainer_cfg.is_multi_task,
            contrastive_max_token_len=trainer_cfg.contrastive_max_token_len,
            should_add_dsts=trainer_cfg.should_add_dsts,
            contrast_with=trainer_cfg.contrast_with,
            single_action_neg_samples=trainer_cfg.single_action_neg_samples,
            train_epochs=trainer_cfg.contrastive_train_epochs,
            contrastive_train_batch_size=trainer_cfg.contrastive_train_batch_size,
        )


class DataModuleConfig:
    def __init__(
        self,
        num_workers=8,
        batch_size=32,
        eval_batch_size=32,
        test_batch_size=32,
        data_split_percent: list[float] = None,
        project_root: str = None,
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        data_prep_out_root: str = "processed_data/simple_tod",
        max_token_len: int = 1024,
        test_prompt_max_len: int = 800,
        num_dialogs: list[int] = None,
        preprocessing_model_name="simple_tod",
        dataset_name="dstc",
        model_name="gpt2",
        tokenizer=None,
        delexicalize: bool = False,
        overwrite: list[bool] = None,
        num_turns: int = 26,
        domain_setting: str = None,
        train_domain_percentage: float = 1.0,
        is_multi_task: bool = False,
        is_multi_head: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_sys_actions: bool = False,
        should_add_user_actions: bool = False,
        single_action_neg_samples: int = 5,
        contrast_with: str = None,
        contrastive_max_token_len: int = 512,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        should_add_dsts: bool = False,
    ):
        self.num_workers = num_workers
        self.preprocessing_model_name = preprocessing_model_name
        self.project_root = Path(project_root)
        self.processed_data_root = self.project_root / data_prep_out_root
        self.raw_data_root = raw_data_root
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.data_split_percent = data_split_percent
        self.max_token_len = max_token_len
        self.test_prompt_max_len = test_prompt_max_len
        self.num_dialogs = num_dialogs
        self.dataset_name = dataset_name
        self.datasets: any = {}
        self.tokenizer = tokenizer or dstc_utils.get_tokenizer()
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False] * len(Steps)
        self.num_turns = num_turns
        self.is_multi_task = is_multi_task
        self.is_multi_head = is_multi_head
        self.multi_tasks = (
            multi_tasks if self.is_multi_task and multi_tasks else [1, 1, 1]
        )
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.train_domain_percentage = train_domain_percentage
        self.domain_setting = domain_setting
        self.domains = DstcDomains[domain_setting.upper()].value
        self.single_action_neg_samples = (
            single_action_neg_samples if single_action_neg_samples else 5
        )
        self.contrast_with = contrast_with
        self.contrastive_max_token_len = contrastive_max_token_len
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results
        self.should_add_dsts = should_add_dsts

    @classmethod
    def from_trainer_config(
        self,
        trainer_config: TrainerConfig,
    ) -> "DataModuleConfig":
        return self(
            num_workers=trainer_config.num_workers,
            project_root=trainer_config.project_root,
            raw_data_root=trainer_config.raw_data_root,
            data_prep_out_root=trainer_config.data_prep_out_root,
            max_token_len=trainer_config.max_token_len,
            test_prompt_max_len=trainer_config.test_prompt_max_len,
            num_dialogs=trainer_config.num_dialogs,
            delexicalize=trainer_config.delexicalize,
            overwrite=trainer_config.overwrite,
            num_turns=trainer_config.num_turns,
            is_multi_head=trainer_config.is_multi_head,
            is_multi_task=trainer_config.is_multi_task,
            multi_tasks=trainer_config.multi_tasks,
            should_add_schema=trainer_config.should_add_schema,
            domain_setting=trainer_config.train_domain_setting,
            train_domain_percentage=trainer_config.train_domain_percentage,
            tokenizer=trainer_config.tokenizer,
            batch_size=trainer_config.train_batch_size,
            eval_batch_size=trainer_config.eval_batch_size,
            test_batch_size=trainer_config.test_batch_size,
            data_split_percent=trainer_config.data_split_percent,
            should_add_user_actions=trainer_config.should_add_user_actions,
            should_add_sys_actions=trainer_config.should_add_sys_actions,
            contrast_with=trainer_config.contrast_with,
            contrastive_max_token_len=trainer_config.contrastive_max_token_len,
            context_type=trainer_config.context_type,
            should_add_service_results=trainer_config.should_add_service_results,
        )

    @classmethod
    def from_inference_config(
        self,
        inf_config: InferenceConfig,
        domain_setting: str = None,
    ) -> "DataModuleConfig":
        return self(
            num_workers=inf_config.num_workers,
            project_root=inf_config.project_root,
            raw_data_root=inf_config.raw_data_root,
            data_prep_out_root=inf_config.data_prep_out_root,
            max_token_len=inf_config.max_token_len,
            test_prompt_max_len=inf_config.test_prompt_max_len,
            num_dialogs=[1, 1, inf_config.num_test_dialogs],
            delexicalize=inf_config.delexicalize,
            overwrite=inf_config.overwrite,
            num_turns=inf_config.num_turns,
            domain_setting=domain_setting,
            train_domain_percentage=inf_config.train_domain_percentage,
            is_multi_head=inf_config.is_multi_head,
            is_multi_task=inf_config.is_multi_task,
            multi_tasks=inf_config.multi_tasks,
            should_add_schema=inf_config.should_add_schema,
            tokenizer=inf_config.tokenizer,
            batch_size=inf_config.test_batch_size,
            eval_batch_size=inf_config.test_batch_size,
            test_batch_size=inf_config.test_batch_size,
            data_split_percent=inf_config.data_split_percent,
            should_add_user_actions=inf_config.should_add_user_actions,
            should_add_sys_actions=inf_config.should_add_sys_actions,
            context_type=inf_config.context_type,
            should_add_service_results=inf_config.should_add_service_results,
        )

    @classmethod
    def from_data_model_exploration(
        self, dme_config: DataModelExplorationConfig
    ) -> "DataModuleConfig":
        return self(
            project_root=dme_config.project_root,
            num_dialogs=dme_config.num_dialogs,
            delexicalize=dme_config.delexicalize,
            overwrite=dme_config.overwrite,
            num_turns=dme_config.num_turns,
            domains=dme_config.domains,
            is_multi_task=dme_config.is_multi_task,
            should_add_schema=dme_config.should_add_schema,
            tokenizer=dme_config.tokenizer,
            data_split_percent=dme_config.data_split_percent,
        )

    @classmethod
    def from_contrastive_config(
        self, c_config: ContrastiveConfig
    ) -> "DataModuleConfig":
        return self(
            project_root=c_config.project_root,
            data_prep_out_root=c_config.data_prep_out_root,
            raw_data_root=c_config.raw_data_root,
            batch_size=c_config.contrastive_train_batch_size,
            eval_batch_size=c_config.eval_batch_size,
            test_batch_size=c_config.test_batch_size,
            num_dialogs=c_config.num_dialogs,
            overwrite=c_config.overwrite,
            num_turns=c_config.num_turns,
            domain_setting=c_config.train_domain_setting,
            is_multi_task=c_config.is_multi_task,
            multi_tasks=c_config.multi_tasks,
            data_split_percent=c_config.data_split_percent,
            should_add_user_actions=c_config.should_add_user_actions,
            single_action_neg_samples=c_config.single_action_neg_samples,
            contrast_with=c_config.contrast_with,
            should_add_dsts=c_config.should_add_dsts,
            contrastive_max_token_len=c_config.contrastive_max_token_len,
        )


class DataPrepConfig:
    def __init__(
        self,
        project_root: str,
        raw_data_root: str,
        processed_data_root: str,
        num_dialogs: list[int] = None,
        delexicalize: bool = True,
        overwrite: list[bool] = None,
        domain_setting: str = None,
        train_domain_percentage: float = 1.0,
        num_turns: int = 26,
        is_multi_task: bool = False,
        is_multi_head: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_sys_actions: bool = False,
        should_add_user_actions: bool = False,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
    ):
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.processed_data_root = self.project_root / processed_data_root
        self.processed_data_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False, False, False]
        self.train_domain_percentage = train_domain_percentage
        self.domain_setting = domain_setting.upper()
        # self.domains = DstcDomains[domain_setting.upper()].value
        self.domains = self._get_domains(self.domain_setting)
        self.num_turns = num_turns
        self.is_multi_task = is_multi_task
        self.is_multi_head = is_multi_head
        self.multi_tasks = (
            multi_tasks if self.is_multi_task and multi_tasks else [1, 1, 1]
        )
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results

    def _get_domains(self, domain_setting: str) -> list[str]:
        domain_to_step_map = {
            DstcDomains.SEEN.name: [Steps.TRAIN.value],
            DstcDomains.UNSEEN.name: [Steps.DEV.value, Steps.TEST.value],
            DstcDomains.ALL.name: [
                Steps.TRAIN.value,
                Steps.DEV.value,
                Steps.TEST.value,
            ],
        }
        step_names = domain_to_step_map[domain_setting]
        if domain_setting == DstcDomains.ALL.name:
            return self._get_domains_from_step_names(step_names)

        used_train_domains, unused_train_domains = self._get_train_domains()
        if domain_setting == DstcDomains.SEEN.name:
            return used_train_domains

        dev_test_domains = self._get_domains_from_step_names(step_names)
        unseen_domains = np.setdiff1d(dev_test_domains, used_train_domains)
        return np.concatenate([unseen_domains, unused_train_domains])

    def _get_train_domains(self):
        domains = np.array(self._get_domains_from_step_names(Steps.TRAIN.value))
        num_choices = int(len(domains) * self.train_domain_percentage)
        random.seed(os.getcwd())
        train_indices = random.sample(range(len(domains)), num_choices)
        mask = np.zeros(len(domains), dtype=bool)
        mask[train_indices] = True
        used_domains = domains[mask]
        unused_domains = domains[~mask]
        return used_domains, unused_domains

    def _get_domains_from_step_names(
        self, step_names: Union[str, list[str]]
    ) -> list[DstcSchema]:
        if isinstance(step_names, str):
            step_names = [step_names]
        schema_strs = np.concatenate(
            [
                utils.read_json(self.raw_data_root / step / "schema.json")
                for step in step_names
            ],
            axis=0,
        )

        schemas = set([DstcSchema.from_dict(schema_str) for schema_str in schema_strs])
        domains = sorted([schema.service_name for schema in schemas])
        # shuffle(domains)
        return domains

    @classmethod
    def from_dm_config(self, dm_config: DataModuleConfig) -> "DataPrepConfig":
        return self(
            project_root=dm_config.project_root,
            raw_data_root=dm_config.raw_data_root,
            processed_data_root=dm_config.processed_data_root,
            num_dialogs=dm_config.num_dialogs,
            delexicalize=dm_config.delexicalize,
            overwrite=dm_config.overwrite,
            num_turns=dm_config.num_turns,
            is_multi_head=dm_config.is_multi_head,
            is_multi_task=dm_config.is_multi_task,
            multi_tasks=dm_config.multi_tasks,
            should_add_schema=dm_config.should_add_schema,
            should_add_sys_actions=dm_config.should_add_sys_actions,
            should_add_user_actions=dm_config.should_add_user_actions,
            domain_setting=dm_config.domain_setting,
            train_domain_percentage=dm_config.train_domain_percentage,
            context_type=dm_config.context_type,
            should_add_service_results=dm_config.should_add_service_results,
        )


class ReconstructDialogConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        out_dir: str = "reconstruct",
        model_path: str = None,
        predictions_dir: str = "./",
    ):
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = (
            self.project_root / model_path if model_path else Path(predictions_dir)
        )
        self.logger = utils.get_logger()
        files = [
            fname
            for fname in os.listdir(self.predictions_dir)
            if fname.endswith(".csv")
        ]
        if not len(files):
            raise ValueError("No csv files found in the model path")
        self.csv_file_names = files

    @classmethod
    def from_inference_config(
        self, t_config: InferenceConfig
    ) -> "ReconstructDialogConfig":
        return self(
            project_root=t_config.project_root,
            raw_data_root=t_config.raw_data_root,
            predictions_dir=os.getcwd(),
        )
