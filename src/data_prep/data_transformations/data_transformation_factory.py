from data_prep.data_transformations.base_data_transformation import (
    BaseDataTransformation,
)
from data_prep.data_transformations.schema_api_transformation import (
    SchemaApiTransformation,
)


class DataTransformationFactory:

    @staticmethod
    def get_data_transformer(
        transformation_name: str, cfg: dict[any, any]
    ) -> BaseDataTransformation:
        if transformation_name == "schema_api_transformation":
            return SchemaApiTransformation(
                current_dataset_path=cfg.raw_data_root,
                original_dataset_path=cfg.project_root / cfg.original_dataset_path,
                step_name=cfg.step_name,
                version_name=cfg.version_name,
                service_results_num_items=cfg.service_results_num_items,
            )
        return ValueError(f"Transformation {transformation_name} not found")
