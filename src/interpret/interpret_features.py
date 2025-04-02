from dataclasses import dataclass

import numpy as np
from my_enums import InterpretFeatureTypes


@dataclass
class FeatureInfo:
    name: str
    domain: str
    feature_type: str

    def get_name(self):
        return self.name

    def get_domain(self):
        return self.domain

    def __init__(self, name: str, domain: str, feature_type: str):
        if feature_type not in InterpretFeatureTypes.__members__.values():
            raise ValueError(
                f"Invalid feature_type: {feature_type}. Must be one of {list(InterpretFeatureTypes)}"
            )
        self.name = name
        self.domain = domain
        self.feature_type = feature_type


@dataclass
class InterpretFeatureGroup:
    features: list[FeatureInfo]
    group_name: str

    def get_feature_types(self):
        return np.unique([f.feature_type for f in self.features])

    def get_feature_names(self):
        return [feature.get_name() for feature in self.features]

    def get_feature_domain_names_pairs(self):
        return [(feature.get_name(), feature.get_domain()) for feature in self.features]
