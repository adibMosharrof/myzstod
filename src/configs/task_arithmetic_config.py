from dataclasses import dataclass
from pathlib import Path
from typing import Union
from base_datamodule import StepData
from my_enums import ContextType, Steps
import utils


@dataclass
class ModelForTaskArithmetic:
    path: Path
    domains: list[str]

    def __init__(self, path: str, domains: list[str], project_root: Path) -> None:
        self.path = project_root / path
        self.domains = domains


class TaskArithmeticConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/data/projects/ZSToD/",
        data_prep_out_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        data_split_percent: float = None,
        num_test_dialogs: int = None,
        out_dir: str = "results",
        wandb: any = None,
        model_a: str = None,
        model_b: str = None,
        model_multi_domain: str = None,
        model_name: str = "distilgpt2",
        train_step_data: StepData = None,
        create_data_from_train: bool = True,
        create_data_from_train_splits: list[float] = None,
        test_batch_size: int = 32,
        postprocess_generation: bool = True,
        quantization: bool = True,
        quantization_dtype: int = 16,
        tokenizer_name: str = None,
        should_add_schema: bool = True,
        should_add_user_actions: bool = True,
        should_add_service_results: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.data_prep_out_root = Path(data_prep_out_root)
        self.data_split_percent = data_split_percent or 1.0
        self.num_test_dialogs = num_test_dialogs or 34
        self.out_dir = Path(out_dir)
        self.raw_data_root = self.project_root / raw_data_root
        self.wandb = wandb
        self.quantization = quantization
        self.quantization_dtype = quantization_dtype
        self.tokenizer_name = tokenizer_name or model_name

        self.tokenizer = utils.get_tokenizer(self.tokenizer_name)
        self.model_a = ModelForTaskArithmetic(project_root=self.project_root, **model_a)
        self.model_b = ModelForTaskArithmetic(project_root=self.project_root, **model_b)
        self.model_multi_domain = ModelForTaskArithmetic(
            project_root=self.project_root, **model_multi_domain
        )
        self.model_name = model_name
        self.train_step_data = StepData(Steps.TRAIN.value, **train_step_data)
        self.create_data_from_train = create_data_from_train
        self.create_data_from_train_splits = create_data_from_train_splits
        self.test_batch_size = test_batch_size
        self.postprocess_generation = postprocess_generation
        self.should_add_schema = should_add_schema
        self.should_add_user_actions = should_add_user_actions
        self.should_add_service_results = should_add_service_results
