from pathlib import Path

from configs.dm_config import DataModuleConfig
from dstc.dstc_domains import DstcDomainBuilder
from multi_head.mh_dataclasses import MultiHeadDictFactory
from my_enums import ContextType, Steps
from accelerate import Accelerator


class DataPrepConfig:
    def __init__(
        self,
        project_root: str,
        raw_data_root: str,
        processed_data_root: str,
        num_dialogs: int = 1,
        delexicalize: bool = False,
        overwrite: bool = False,
        domain_setting: str = None,
        train_domain_percentage: float = 1.0,
        num_turns: int = 26,
        is_scale_grad: bool = False,
        is_multi_task: bool = False,
        is_multi_head: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_sys_actions: bool = False,
        should_add_user_actions: bool = False,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        service_results_num_items: int = 1,
        mh_fact: MultiHeadDictFactory = None,
        data_prep_multi_process: bool = True,
        step_name: str = Steps.TRAIN.value,
        original_dataset_path: str = None,
        data_prep_transformations: list[str] = None,
        data_augmentations: list[str] = None,
        version_name: str = None,
        dataset_name: str = None,
        **kwargs,
    ):
        self.accelerator = Accelerator()
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.processed_data_root = self.project_root / processed_data_root
        self.processed_data_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        self.delexicalize = delexicalize
        self.overwrite = overwrite
        self.train_domain_percentage = train_domain_percentage
        self.domain_setting = domain_setting
        self.domains = DstcDomainBuilder(
            self.raw_data_root, train_domain_percentage
        ).get_domains(self.domain_setting)
        # self.domains = None
        self.num_turns = num_turns
        self.is_scale_grad = is_scale_grad
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
        self.mh_fact = mh_fact if mh_fact else None
        self.data_prep_multi_process = data_prep_multi_process
        self.step_name = step_name
        self.service_results_num_items = service_results_num_items
        self.original_dataset_path = original_dataset_path
        self.data_prep_transformations = data_prep_transformations
        self.data_augmentations = data_augmentations
        self.version_name = version_name
        self.dataset_name = dataset_name

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
            is_scale_grad=dm_config.is_scale_grad,
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
            mh_fact=dm_config.mh_fact,
            data_prep_multi_process=dm_config.data_prep_multi_process,
            step_name=dm_config.step_name,
            service_results_num_items=dm_config.service_results_num_items,
            original_dataset_path=dm_config.original_dataset_path,
            data_prep_transformations=dm_config.data_prep_transformations,
            data_augmentations=dm_config.data_augmentations,
            version_name=dm_config.version_name,
            dataset_name=dm_config.dataset_name,
        )
