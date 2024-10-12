from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Union
from dstc.dstc_domains import DstcDomains
from multi_head.mh_dataclasses import MultiHeadDictFactory
from my_enums import ContextType, MultiTaskNames, Steps
import dstc.dstc_utils as dstc_utils
from accelerate import Accelerator

if TYPE_CHECKING:
    from configs.trainer_config import TrainerConfig
    from configs.inference_config import InferenceConfig
    from configs.data_model_exploration_config import DataModelExplorationConfig
    from configs.dm_config import DataModuleConfig
    from configs.contrastive_config import ContrastiveConfig
    from base_datamodule import StepData


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
        train_domain_settings: Union[list[str], str] = None,
        dev_domain_settings: Union[list[str], str] = None,
        test_domain_settings: Union[list[str], str] = None,
        train_domain_percentage: float = 1.0,
        create_data_from_train: bool = False,
        create_data_from_train_splits: list[float] = None,
        is_scale_grad: bool = False,
        is_multi_task: bool = False,
        is_multi_head: bool = False,
        multi_tasks: list[MultiTaskNames] = None,
        should_add_schema: bool = False,
        should_add_sys_actions: bool = False,
        should_add_user_actions: bool = False,
        single_action_neg_samples: int = 5,
        contrast_with: str = None,
        contrastive_max_token_len: int = 512,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        should_add_dsts: bool = False,
        mh_fact: MultiHeadDictFactory = None,
        data_prep_multi_process: bool = True,
        test_num_turns_groups: list[Tuple[int, int]] = None,
        train_step_data: "StepData" = None,
        accelerator: "Accelerator" = None,
        service_results_num_items: int = 1,
        original_dataset_path: str = None,
        data_prep_transformations: list[str] = None,
        data_augmentations: list[any] = None,
        version_name: str = None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.accelerator = Accelerator()
        self.num_workers = num_workers
        self.model_name = model_name
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
        self.tokenizer = tokenizer or dstc_utils.get_tokenizer(model_name)
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False] * len(Steps)
        self.num_turns = num_turns
        self.is_scale_grad = is_scale_grad
        self.is_multi_task = is_multi_task
        self.is_multi_head = is_multi_head
        self.multi_tasks = (
            multi_tasks if self.is_multi_task and multi_tasks else MultiTaskNames.list()
        )
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.train_domain_percentage = train_domain_percentage
        self.single_action_neg_samples = (
            single_action_neg_samples if single_action_neg_samples else 5
        )
        if self.is_scale_grad and not self.should_add_schema:
            raise ValueError(
                "is_scale_grad is true but should_add_schema is false, which is not allowed"
            )
        self.contrast_with = contrast_with
        self.contrastive_max_token_len = contrastive_max_token_len
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results
        self.should_add_dsts = should_add_dsts
        self.mh_fact = (
            mh_fact
            if mh_fact
            else MultiHeadDictFactory(self.tokenizer) if is_multi_head else None
        )
        self.data_prep_multi_process = data_prep_multi_process
        self.train_domain_settings = train_domain_settings
        self.dev_domain_settings = dev_domain_settings
        self.test_domain_settings = test_domain_settings
        self.create_data_from_train = create_data_from_train
        self.create_data_from_train_splits = create_data_from_train_splits or [0.1, 0.1]
        # these two variables are added so that we can have typing in DataPrepConfig.from_dm_config method
        self.step_name = None
        self.domain_setting = None
        self.test_num_turns_groups = test_num_turns_groups
        self.train_step_data = train_step_data
        self.service_results_num_items = service_results_num_items
        self.original_dataset_path = original_dataset_path
        self.data_prep_transformations = data_prep_transformations
        self.data_augmentations = data_augmentations
        self.version_name = version_name

    @classmethod
    def from_trainer_config(
        self,
        trainer_config: "TrainerConfig",
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
            model_name=trainer_config.model_name,
            is_scale_grad=trainer_config.is_scale_grad,
            is_multi_head=trainer_config.is_multi_head,
            is_multi_task=trainer_config.is_multi_task,
            multi_tasks=trainer_config.multi_tasks,
            should_add_schema=trainer_config.should_add_schema,
            train_domain_settings=trainer_config.train_domain_settings,
            dev_domain_settings=trainer_config.dev_domain_settings,
            test_domain_settings=trainer_config.test_domain_settings,
            train_domain_percentage=trainer_config.train_domain_percentage,
            create_data_from_train=trainer_config.create_data_from_train,
            create_data_from_train_splits=trainer_config.create_data_from_train_splits,
            tokenizer=trainer_config.tokenizer,
            batch_size=trainer_config.train_batch_size,
            eval_batch_size=trainer_config.eval_batch_size,
            test_batch_size=trainer_config.test_batch_size,
            data_split_percent=trainer_config.data_split_percent,
            should_add_user_actions=trainer_config.should_add_user_actions,
            should_add_sys_actions=trainer_config.should_add_sys_actions,
            contrast_with=trainer_config.contrast_with,
            contrastive_max_token_len=trainer_config.contrastive_max_token_len,
            context_type=trainer_config.model_type.context_type,
            should_add_service_results=trainer_config.should_add_service_results,
            mh_fact=trainer_config.mh_fact,
            data_prep_multi_process=trainer_config.data_prep_multi_process,
            test_num_turns_groups=trainer_config.test_num_turns_groups,
            service_results_num_items=trainer_config.service_results_num_items,
        )

    @classmethod
    def from_inference_config(
        self,
        inf_config: "InferenceConfig",
        domain_setting: str = None,
        train_step_data: "StepData" = None,
    ) -> "DataModuleConfig":
        return self(
            num_workers=inf_config.num_workers,
            project_root=inf_config.project_root,
            raw_data_root=inf_config.raw_data_root,
            data_prep_out_root=inf_config.data_prep_out_root,
            model_name=inf_config.model_name,
            max_token_len=inf_config.max_token_len,
            test_prompt_max_len=inf_config.test_prompt_max_len,
            num_dialogs=[inf_config.num_train_dialogs, 1, inf_config.num_test_dialogs],
            delexicalize=inf_config.delexicalize,
            overwrite=inf_config.overwrite,
            num_turns=inf_config.num_turns,
            test_domain_settings=domain_setting,
            train_domain_percentage=inf_config.train_domain_percentage,
            create_data_from_train=inf_config.create_data_from_train,
            create_data_from_train_splits=inf_config.create_data_from_train_splits,
            is_scale_grad=inf_config.is_scale_grad,
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
            mh_fact=inf_config.mh_fact,
            data_prep_multi_process=inf_config.data_prep_multi_process,
            test_num_turns_groups=inf_config.test_num_turns_groups,
            train_step_data=train_step_data,
        )

    @classmethod
    def from_data_model_exploration(
        self, dme_config: "DataModelExplorationConfig"
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
        self, c_config: "ContrastiveConfig"
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
            train_domain_settings=c_config.train_domain_settings,
            dev_domain_settings=c_config.dev_domain_settings,
            test_domain_settings=c_config.test_domain_settings,
            is_multi_task=c_config.is_multi_task,
            multi_tasks=c_config.multi_tasks,
            data_split_percent=c_config.data_split_percent,
            should_add_user_actions=c_config.should_add_user_actions,
            single_action_neg_samples=c_config.single_action_neg_samples,
            contrast_with=c_config.contrast_with,
            should_add_dsts=c_config.should_add_dsts,
            contrastive_max_token_len=c_config.contrastive_max_token_len,
            data_prep_multi_process=c_config.data_prep_multi_process,
        )
