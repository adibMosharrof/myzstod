from utilities.dialog_studio_dataclasses import DsDialog
from data_prep.data_transformations.base_data_transformation import (
    BaseDataTransformation,
)
from pathlib import Path
from sgd_dstc8_data_model.dstc_dataclasses import (
    get_schemas,
)


class SchemaApiTransformation(BaseDataTransformation):
    def __init__(
        self,
        current_dataset_path: str,
        original_dataset_path: str,
        step_name: str,
        version_name: str,
        service_results_num_items: int,
    ):
        self.current_schemas = get_schemas(current_dataset_path, step_name)
        self.original_schemas = get_schemas(original_dataset_path, step_name)
        self.service_results_num_items = service_results_num_items
        self.version_name = version_name
        self.schema_update_map = {}

    def transform(self, dialog: DsDialog) -> DsDialog:
        for log in dialog.log:
            if not log.original_system_side_information:
                continue
            if not log.original_user_side_information:
                continue
            for frame in log.original_user_side_information.frames:
                frame.service = self.get_new_service_name(frame.service)

            for frame in log.original_system_side_information.frames:
                original_service = frame.service
                frame.service = self.get_new_service_name(frame.service)
                if not frame.service_call:
                    continue
                schema_map = self.get_schema_update_map(original_service, frame.service)
                frame.service_call.method = schema_map.get(frame.service_call.method)
                params = frame.service_call.parameters
                param_names = list(params.keys())
                for param_name in param_names:
                    params[schema_map.get(param_name)] = params.pop(param_name)
                if not frame.service_results:
                    continue
                for result in frame.service_results[: self.service_results_num_items]:
                    result_keys = list(result.keys())
                    for key in result_keys:
                        result[schema_map.get(key)] = result.pop(key)
        return dialog

    def get_new_service_name(self, old_service_name: str) -> str:
        return old_service_name + str(self.version_name)

    def get_new_schema(self, old_schema_name: str) -> str:
        return self.current_schemas.get(old_schema_name)

    def get_schema_update_map(
        self, original_service: str, new_service: str
    ) -> dict[str, str]:
        if original_service in self.schema_update_map:
            return self.schema_update_map[original_service]
        schema_orig_to_new_map = {}
        orig_schema = self.original_schemas.get(original_service)
        new_schema = self.current_schemas.get(new_service)
        if not orig_schema:
            raise ValueError(
                f"Original Schema not found for service {original_service}"
            )
        if not new_schema:
            raise ValueError(f"New Schema not found for service {new_service}")
        for orig_slot, new_slot in zip(orig_schema.slots, new_schema.slots):
            schema_orig_to_new_map[orig_slot.name] = new_slot.name
        for orig_intent, new_intent in zip(orig_schema.intents, new_schema.intents):
            schema_orig_to_new_map[orig_intent.name] = new_intent.name
        self.schema_update_map[original_service] = schema_orig_to_new_map
        return schema_orig_to_new_map
