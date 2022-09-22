from lib2to3.pgen2.tokenize import tokenize
import os
from pathlib import Path
from typing import Dict
from datasets import Dataset

from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2PreTrainedModel

from my_enums import DstcDomains, SpecialTokens, Steps
import dstc_utils
import utils
import re

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
        domains: list[str] = None,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        test_settings: list[str] = None,
        out_dir: str = "results",
        tokenizer: AutoTokenizer = None,
        context_max_len: int = 600,
        target_max_len: int = 424,
        is_multi_task: bool = False,
        should_add_schema: bool = False,
    ) -> None:
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 0.1]
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
        self.domains = domains or [
            "Buses",
            "Events",
            "Flights",
            "Homes",
            "Hotels",
            "Media",
            "Movies",
            "Music",
            "RentalCars",
            "Restaurants",
            "RideSharing",
            "Services",
            "Travel",
            "Weather",
        ]
        self.test_settings = test_settings or ["seen"]
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.out_dir = out_dir
        self.tokenizer = tokenizer
        self.context_max_len = context_max_len
        self.target_max_len = target_max_len
        self.predictions_log_dir = Path(predictions_log_dir)
        self.predictions_log_dir.mkdir(parents=True, exist_ok=True)
        self.is_multi_task = is_multi_task
        self.should_add_schema = should_add_schema
        self.logger = utils.get_logger()
        self.tokenizer = (
            self.tokenizer
            if self.tokenizer
            else self._get_tokenizer(model)
        )
        self.padding_regexp = re.compile(re.escape(SpecialTokens.pad_token))

    def _get_tokenizer(self, model_path_str:str):
        model_path:Path = self.project_root / model_path_str
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
        max_token_len: int = 512,
        num_dialogs: list[int] = None,
        delexicalize: bool = False,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_settings: str = "SEEN",
        test_settings: list[str] = None,
        train_settings: str = "seen",
        output_dir: str = "results",
        pretrain_epochs: int = 1,
        pretrain_model_path: str = None,
        train_epochs: int = 1,
        logging_dir: str = "logs",
        generate_max_len: int = 1024,
        domains: list[str] = None,
        should_test: bool = False,
        logging_steps: int = 50,
        context_max_len: int = 800,
        target_max_len: int = 224,
        eval_accumulation_steps: int = 25,
        is_multi_task: bool = False,
        should_add_schema: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.data_prep_out_root = Path(data_prep_out_root)
        self.model_name = model_name
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.max_token_len = max_token_len
        self.num_dialogs = num_dialogs or [127, 20, 34]
        self.delexicalize = delexicalize
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.test_settings = test_settings or ["seen"]
        self.output_dir = Path(output_dir)
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.train_settings = train_settings
        self.pretrain_model_path = pretrain_model_path
        self.logging_dir = Path(logging_dir)
        self.generate_max_len = generate_max_len
        self.domains = (
            domains if domains else DstcDomains[train_domain_settings.upper()].value
        )
        self.should_test = should_test
        self.delexicalize = delexicalize
        self.logging_steps = logging_steps
        self.train_batch_size = train_batch_size
        self.raw_data_root = raw_data_root
        self.context_max_len = context_max_len
        self.target_max_len = target_max_len
        self.eval_accumulation_steps = eval_accumulation_steps
        self.is_multi_task = is_multi_task
        self.tokenizer = dstc_utils.get_tokenizer(model_name)
        self.should_add_schema = should_add_schema


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
        max_token_len: int = 128,
        num_dialogs: list[int] = None,
        preprocessing_model_name="simple_tod",
        dataset_name="dstc",
        model_name="gpt2",
        tokenizer=None,
        delexicalize: bool = False,
        overwrite: list[bool] = None,
        num_turns: int = 26,
        domains: list[str] = None,
        is_multi_task: bool = False,
        should_add_schema: bool = False,
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
        self.num_dialogs = num_dialogs
        self.dataset_name = dataset_name

        self.datasets: Dict[str, Dataset] = {}
        self.tokenizer = tokenizer or dstc_utils.get_tokenizer()
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False] * len(Steps)
        self.num_turns = num_turns
        self.domains = domains or ["restaurant", "hotel", "attraction", "train"]
        self.is_multi_task = is_multi_task
        self.should_add_schema = should_add_schema

    @classmethod
    def from_trainer_config(self, trainer_config: TrainerConfig) -> "DataModuleConfig":
        return self(
            num_workers=trainer_config.num_workers,
            project_root=trainer_config.project_root,
            raw_data_root=trainer_config.raw_data_root,
            data_prep_out_root=trainer_config.data_prep_out_root,
            max_token_len=trainer_config.max_token_len,
            num_dialogs=trainer_config.num_dialogs,
            delexicalize=trainer_config.delexicalize,
            overwrite=trainer_config.overwrite,
            num_turns=trainer_config.num_turns,
            domains=trainer_config.domains,
            is_multi_task=trainer_config.is_multi_task,
            should_add_schema=trainer_config.should_add_schema,
            tokenizer=trainer_config.tokenizer,
            batch_size=trainer_config.train_batch_size,
            eval_batch_size=trainer_config.eval_batch_size,
            test_batch_size=trainer_config.test_batch_size,
            data_split_percent=trainer_config.data_split_percent,
        )

    @classmethod
    def from_inference_config(self, inf_config:InferenceConfig) ->'DataModuleConfig':
        return self(
            num_workers=inf_config.num_workers,
            project_root=inf_config.project_root,
            raw_data_root=inf_config.raw_data_root,
            data_prep_out_root=inf_config.data_prep_out_root,
            max_token_len=inf_config.max_token_len,
            num_dialogs=[1,1,inf_config.num_test_dialogs],
            delexicalize=inf_config.delexicalize,
            overwrite=[False, False, True],
            num_turns=inf_config.num_turns,
            domains=inf_config.domains,
            is_multi_task=inf_config.is_multi_task,
            should_add_schema=inf_config.should_add_schema,
            tokenizer=inf_config.tokenizer,
            batch_size=inf_config.test_batch_size,
            eval_batch_size=inf_config.test_batch_size,
            test_batch_size=inf_config.test_batch_size,
            data_split_percent=inf_config.data_split_percent,
        )


class DataPrepConfig:
    def __init__(
        self,
        project_root: str,
        data_root: str,
        out_root: str,
        num_dialogs: list[int] = None,
        delexicalize: bool = True,
        overwrite: list[bool] = None,
        domains: list[str] = None,
        num_turns: int = 55,
        is_multi_task: bool = False,
        should_add_schema: bool = False,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = self.project_root / out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False, False, False]
        self.domains = domains or ["Buses", "Hotels", "Events"]
        self.num_turns = num_turns
        self.is_multi_task = is_multi_task
        self.should_add_schema = should_add_schema

    @classmethod
    def from_dm_config(self, dm_config: DataModuleConfig) -> "DataPrepConfig":
        return self(
            dm_config.project_root,
            dm_config.raw_data_root,
            dm_config.processed_data_root,
            dm_config.num_dialogs,
            dm_config.delexicalize,
            dm_config.overwrite,
            dm_config.domains,
            dm_config.num_turns,
            dm_config.is_multi_task,
            dm_config.should_add_schema,
        )


class ReconstructDialogConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/projects/generative_tod/",
        raw_data_root: str="data/dstc8-schema-guided-dialogue/",
        out_dir: str = "results",
        model_path: str = None,
    ):
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.project_root/model_path
        self.logger = utils.get_logger()
        files = [fname for fname in os.listdir(self.model_path) if fname.endswith(".csv")]
        if not len(files):
            raise ValueError("No csv files found in the model path")
        self.predictions_csv_path = self.model_path / files[0]
        