import os
import re
from lib2to3.pgen2.tokenize import tokenize
from pathlib import Path
from typing import Dict

from datasets import Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2PreTrainedModel

import dstc_utils
import utils
from my_enums import DstcDomains, SpecialTokens, Steps


class TrainerConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        data_prep_out_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        model_name: str = "gpt2",
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 32,
        train_batch_size: int = 8,
        num_dialogs: list[int] = None,
        delexicalize: bool = False,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_setting: str = "SEEN",
        test_domain_settings: list[str] = None,
        out_dir: str = "results",
        pretrain_epochs: int = 1,
        pretrain_model_path: str = None,
        train_epochs: int = 1,
        logging_dir: str = "logs",
        generate_max_len: int = 1024,
        should_test: bool = False,
        logging_steps: int = 50,
        test_prompt_max_len: int = 799,
        max_token_len: int = 1022,
        eval_accumulation_steps: int = 5,
        is_multi_task: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_user_actions: bool = False,
        should_add_sys_actions: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.data_prep_out_root = Path(data_prep_out_root)
        self.model_name = model_name
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.max_token_len = max_token_len
        self.num_dialogs = num_dialogs or [20, 10, 17]
        self.delexicalize = delexicalize
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.test_domain_settings = test_domain_settings or ["all", "seen", "unseen"]
        self.out_dir = Path(out_dir)
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.train_domain_setting = train_domain_setting
        self.pretrain_model_path = pretrain_model_path
        self.logging_dir = Path(logging_dir)
        self.generate_max_len = generate_max_len
        self.should_test = should_test
        self.delexicalize = delexicalize
        self.logging_steps = logging_steps
        self.train_batch_size = train_batch_size
        self.raw_data_root = raw_data_root
        self.test_prompt_max_len = test_prompt_max_len
        self.eval_accumulation_steps = eval_accumulation_steps
        self.is_multi_task = is_multi_task
        self.multi_tasks = multi_tasks or [1, 1, 1]
        self.tokenizer = dstc_utils.get_tokenizer(model_name)
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        if test_prompt_max_len > max_token_len:
            raise ValueError("context_max_len must be less than max_token_len")


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
        num_test_dialogs: int = 1,
        delexicalize: bool = False,
        model: str = "outputs/2022-07-26/22-28-09/results/train/checkpoint-7067",
        model_name: str = "gpt2",
        generate_max_len: int = 1024,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        test_domain_settings: list[str] = None,
        out_dir: str = "results",
        tokenizer: AutoTokenizer = None,
        test_prompt_max_len: int = 799,
        is_multi_task: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_user_actions: bool = False,
        should_add_sys_actions: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.max_token_len = max_token_len
        self.raw_data_root = raw_data_root
        self.project_root = Path(project_root)
        self.data_prep_out_root = data_prep_out_root
        self.num_test_dialogs = num_test_dialogs
        self.delexicalize = delexicalize
        self.model = self._get_model(model)
        self.model_name = model_name
        self.generate_max_len = generate_max_len
        self.test_domain_settings = test_domain_settings or ["all", "seen", "unseen"]
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.test_prompt_max_len = test_prompt_max_len
        self.predictions_log_dir = Path(predictions_log_dir)
        self.predictions_log_dir.mkdir(parents=True, exist_ok=True)
        self.is_multi_task = is_multi_task
        self.multi_tasks = multi_tasks or [1, 1, 1]
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.logger = utils.get_logger()
        self.tokenizer = (
            self.tokenizer if self.tokenizer else self._get_tokenizer(model)
        )
        self.padding_regexp = re.compile(re.escape(SpecialTokens.pad_token))

    def _get_tokenizer(self, model_path_str: str):
        model_path: Path = self.project_root / model_path_str
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
        is_multi_task: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_sys_actions: bool = False,
        should_add_user_actions: bool = False,
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
        self.datasets: Dict[str, Dataset] = {}
        self.tokenizer = tokenizer or dstc_utils.get_tokenizer()
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False] * len(Steps)
        self.num_turns = num_turns
        self.is_multi_task = is_multi_task
        self.multi_tasks = multi_tasks or [1, 1, 1]
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.domain_setting = domain_setting
        self.domains = DstcDomains[domain_setting.upper()].value

    @classmethod
    def from_trainer_config(self, trainer_config: TrainerConfig) -> "DataModuleConfig":
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
            is_multi_task=trainer_config.is_multi_task,
            multi_tasks=trainer_config.multi_tasks,
            should_add_schema=trainer_config.should_add_schema,
            domain_setting=trainer_config.train_domain_setting,
            tokenizer=trainer_config.tokenizer,
            batch_size=trainer_config.train_batch_size,
            eval_batch_size=trainer_config.eval_batch_size,
            test_batch_size=trainer_config.test_batch_size,
            data_split_percent=trainer_config.data_split_percent,
            should_add_user_actions=trainer_config.should_add_user_actions,
            should_add_sys_actions=trainer_config.should_add_sys_actions,
        )

    @classmethod
    def from_inference_config(
        self,
        inf_config: InferenceConfig,
        domain_setting: str = None,
        prompt_token_map: dict = None,
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


class DataPrepConfig:
    def __init__(
        self,
        project_root: str,
        data_root: str,
        processed_data_root: str,
        num_dialogs: list[int] = None,
        delexicalize: bool = True,
        overwrite: list[bool] = None,
        domain_setting: str = None,
        num_turns: int = 26,
        is_multi_task: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_sys_actions: bool = False,
        should_add_user_actions: bool = False,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.processed_data_root = self.project_root / processed_data_root
        self.processed_data_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False, False, False]
        self.domain_setting = domain_setting
        self.domains = DstcDomains[domain_setting.upper()].value
        self.num_turns = num_turns
        self.is_multi_task = is_multi_task
        self.multi_tasks = multi_tasks or [1, 1, 1]
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions

    @classmethod
    def from_dm_config(self, dm_config: DataModuleConfig) -> "DataPrepConfig":
        return self(
            project_root=dm_config.project_root,
            data_root=dm_config.raw_data_root,
            processed_data_root=dm_config.processed_data_root,
            num_dialogs=dm_config.num_dialogs,
            delexicalize=dm_config.delexicalize,
            overwrite=dm_config.overwrite,
            num_turns=dm_config.num_turns,
            is_multi_task=dm_config.is_multi_task,
            multi_tasks=dm_config.multi_tasks,
            should_add_schema=dm_config.should_add_schema,
            should_add_sys_actions=dm_config.should_add_sys_actions,
            should_add_user_actions=dm_config.should_add_user_actions,
            domain_setting=dm_config.domain_setting,
        )


class ReconstructDialogConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        out_dir: str = "results",
        model_path: str = None,
    ):
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.project_root / model_path
        self.logger = utils.get_logger()
        files = [
            fname for fname in os.listdir(self.model_path) if fname.endswith(".csv")
        ]
        if not len(files):
            raise ValueError("No csv files found in the model path")
        self.csv_file_names = files


class ConstrastiveConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        data_prep_out_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        model_name: str = "bert",
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 32,
        train_batch_size: int = 8,
        num_dialogs: list[int] = None,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_setting: str = "SEEN",
        test_domain_settings: list[str] = None,
        out_dir: str = "results",
        train_epochs: int = 1,
        logging_dir: str = "logs",
        logging_steps: int = 50,
        eval_accumulation_steps: int = 5,
        is_multi_task: bool = False,
        multi_tasks: list[int] = None,
        should_add_user_actions: bool = False,
        should_add_sys_actions: bool = False,
    ):
        self.project_root = Path(project_root)
        self.data_prep_out_root = self.project_root / data_prep_out_root
        self.raw_data_root = Path(raw_data_root)
        self.model_name = model_name
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.num_dialogs = num_dialogs
        self.num_turns = num_turns
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
        self.should_add_user_actions = should_add_user_actions
        self.should_add_sys_actions = should_add_sys_actions
