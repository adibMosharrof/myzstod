from datamodules.data_filters.api_in_context_filter import (
    ApiInContextFilter,
)
from datamodules.data_augmentation.pseudo_label_augmentation import (
    PseudoLabelAugmentation,
)
from schema.schema_loader import SchemaLoader
from enum import Enum

from tod.turns.api_call_turn_csv_row import ApiCallTurnCsvRow


class AugmentationNames(str, Enum):
    PSEUDO_LABEL_AUGMENTATION = "pseudo_labels"


class DataAugmentationFactory:

    @classmethod
    def create_data_augmentations(self, cfg, schemas):
        augmentations = {}
        aug_cfgs = cfg.get("data_augmentations", None)
        if aug_cfgs is None:
            return augmentations
        for aug_name, aug_params in aug_cfgs.items():
            if aug_name == AugmentationNames.PSEUDO_LABEL_AUGMENTATION.value:
                augmentations[aug_name] = PseudoLabelAugmentation(
                    cfg,
                    **aug_params,
                    schemas=schemas,
                    turn_row_csv_cls=ApiCallTurnCsvRow()
                )

        return augmentations
