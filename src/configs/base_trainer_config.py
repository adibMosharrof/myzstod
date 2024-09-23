from dataclasses import dataclass, field
import logging
import os
from typing import Union, Optional
from pathlib import Path


@dataclass
class BaseTrainerConfig:
    num_workers: int = 8
    test_prompt_max_len: int = 700
    max_token_len: int = 1024
    num_turns: int = 26
    should_add_schema: bool = True
    should_add_user_actions: bool = False
    should_add_service_results: bool = True
    service_results_num_items: int = 2
    early_stopping_patience: int = 2
    project_root: str = "/mounts/u-amo-d1/adibm-data/projects/ZSToD"
    resume_checkpoint: str = ""
    epochs: int = 1
    is_scale_grad: bool = False
    tokenizer_name: str = None
    train_domain_settings: list[str] = field(default_factory=lambda: ["Restaurants_1"])
    dev_domain_settings: list[str] = field(default_factory=lambda: ["Restaurants_2"])
    test_domain_settings: list[list[str]] = field(
        default_factory=lambda: [["Restaurants_2"]]
    )
    prompt_type: str = "default"
    overwrite: list[int] = field(default_factory=lambda: [0, 0, 0])
    out_dir: str = "results"
    should_train: bool = True
    should_test: bool = True
    wandb: dict = field(
        default_factory=lambda: {
            "project": "ZSTod",
            "entity": "adibm",
            "notes": "sgd",
            "task": "probing",
        }
    )
    hydra_run_dir: str = "./outputs/probing"
    data_size: dict[str, any] = None
    dataset: list[dict[str, any]] = None
    model_type: dict[str, any] = None

    def __post_init__(self):
        # Convert paths to Path objects
        self.hydra_run_dir = Path(self.hydra_run_dir)
        self.project_root = Path(self.project_root)
        self.out_dir = str(os.getcwd() / Path(self.out_dir))

        # Initialize Dataset and Model configurations
        if self.dataset is not None:
            self.dataset = {
                dataset_name: DatasetConfig(**dataset_config)
                for dataset_name, dataset_config in self.dataset.items()
            }

        # Model type and data size configuration
        model_type_cfg = ModelTypeConfig(**self.model_type)
        data_size_cfg = DataSizeConfig(**self.data_size)

        self.model_name = model_type_cfg.model_log_name
        self.model_path = model_type_cfg.model_path
        self.context_type = model_type_cfg.context_type
        self.train_batch_size = model_type_cfg.train_batch_size
        self.eval_batch_size = model_type_cfg.eval_batch_size
        self.test_batch_size = model_type_cfg.test_batch_size
        self.gradient_accumulation_steps = model_type_cfg.gradient_accumulation_steps
        self.eval_accumulation_steps = model_type_cfg.eval_accumulation_steps
        self.quantization = model_type_cfg.quantization
        self.quantization_dtype = model_type_cfg.quantization_dtype
        self.learning_rate = model_type_cfg.learning_rate
        self.model_log_name = model_type_cfg.model_log_name

        self.data_split_percent = data_size_cfg.data_split_percent
        self.num_dialogs = data_size_cfg.num_dialogs
        self.eval_steps = data_size_cfg.eval_steps
        self.save_steps = data_size_cfg.save_steps

        # Initialize logger
        formatter = logging.Formatter(fmt="%(message)s")
        root_logger = logging.getLogger()  # no name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
        self.logger = root_logger


@dataclass
class ModelTypeConfig:
    model_name: str = ""
    model_path: str = ""
    train_batch_size: int = 5
    eval_batch_size: int = 10
    test_batch_size: int = 70
    gradient_accumulation_steps: int = 32
    eval_accumulation_steps: int = 32
    quantization: bool = False
    quantization_dtype: int = 16
    learning_rate: float = 1e-3
    model_log_name: str = "gpt2"
    context_type: str = "gpt_api_call"


@dataclass
class DataSizeConfig:
    data_split_percent: list[int] = field(default_factory=lambda: [1, 1, 1])
    num_dialogs: list[int] = field(default_factory=lambda: [1, 1, 1])
    save_steps: int = 1
    eval_steps: int = 1


@dataclass
class DatasetConfig:
    raw_data_root: str
    data_prep_out_root: str
    chatgpt_results_path: str
    dataset_name: str
