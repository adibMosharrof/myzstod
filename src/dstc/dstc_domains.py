from enum import Enum
import os
from pathlib import Path
import random
from typing import Union

import numpy as np
from omegaconf import ListConfig
from dstc.dstc_dataclasses import DstcSchema
from my_enums import Steps
import utils


class DstcDomains(str, Enum):
    # make sure the string values exactly match the domain names in the data
    SEEN = "seen"
    UNSEEN = "unseen"
    ALL = "all"
    # TRAVEL_1 = "Travel_1"
    # RESTAURANTS_2 = "Restaurants_2"
    # RESTAURANTS_1 = "Restaurants_1"
    # HOTELS_2 = "Hotels_2"
    # MOVIES_3 = "Movies_3"
    # RENTALCARS_1 = "RentalCars_1"

    @classmethod
    def regular_settings(cls):
        return [cls.SEEN.value, cls.UNSEEN.value, cls.ALL.value]

    def __getitem__(cls, name):
        return cls._member_map_[name]

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class DstcDomainBuilder:
    def __init__(self, raw_data_root: Path, train_domain_percentage: float) -> None:
        self.raw_data_root = raw_data_root
        self.train_domain_percentage = train_domain_percentage

    def get_domains(self, domain_setting: Union[str, list[str]]) -> list[str]:
        if isinstance(domain_setting, (list, ListConfig)):
            out = []
            for domain in domain_setting:
                out.append(self.get_domains(domain))
            return np.concatenate(out, axis=0)
        if domain_setting not in DstcDomains.regular_settings():
            return [domain_setting]

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
