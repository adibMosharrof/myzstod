

from pathlib import Path
from configs.contrastive_config import ContrastiveConfig
from configs.data_model_exploration_config import DataModelExplorationConfig
from configs.inference_config import InferenceConfig
from configs.trainer_config import TrainerConfig
from multi_head.mh_dataclasses import MultiHeadDictFactory
from my_enums import ContextType, DstcDomains, Steps
import dstc_utils

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
        mh_fact: MultiHeadDictFactory = None,
        data_prep_multi_process: bool = True,
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
        self.mh_fact = (
            mh_fact
            if mh_fact
            else MultiHeadDictFactory(self.tokenizer)
            if is_multi_head
            else None
        )
        self.data_prep_multi_process = data_prep_multi_process

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
            mh_fact=trainer_config.mh_fact,
            data_prep_multi_process=trainer_config.data_prep_multi_process,
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
            mh_fact=inf_config.mh_fact,
            data_prep_multi_process=inf_config.data_prep_multi_process,
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
            data_prep_multi_process = c_config.data_prep_multi_process,
        )

