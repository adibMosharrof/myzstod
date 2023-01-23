
import os
from pathlib import Path
import random
from typing import Union

import numpy as np
from configs.dm_config import DataModuleConfig
from dstc_dataclasses import DstcSchema
from multi_head.mh_dataclasses import MultiHeadDictFactory
from my_enums import ContextType, DstcDomains, Steps
import utils

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
        mh_fact: MultiHeadDictFactory = None,
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
        self.mh_fact = mh_fact if mh_fact else None

    def _get_domains(self, domain_setting: str) -> list[str]:
        if domain_setting not in DstcDomains.regular_settings():
            return DstcDomains[domain_setting].value

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
            mh_fact=dm_config.mh_fact,
        )
