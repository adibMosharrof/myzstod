from datamodules.data_augmentation.pseudo_label_augmentation import (
    PseudoLabelAugmentation,
)
from schema.schema_loader import SchemaLoader
from enum import Enum


class AugmentationNames(str, Enum):
    PSEUDO_LABEL_AUGMENTATION = "pseudo_label_augmentation"


class DataAugmentationFactory:

    @classmethod
    def create_data_augmentations(self, cfg, schemas):
        augmentations = []
        aug_cfgs = cfg.get("data_augmentations", None)
        if aug_cfgs is None:
            return augmentations
        for aug_cfg in aug_cfgs:
            for name, params in aug_cfg.items():
                if name == AugmentationNames.PSEUDO_LABEL_AUGMENTATION.value:
                    augmentations.append(
                        PseudoLabelAugmentation(cfg, **params, schemas=schemas)
                    )
        return augmentations
