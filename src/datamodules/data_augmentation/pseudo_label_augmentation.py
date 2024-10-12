from dotmap import DotMap

from schema.pseudo_schema_dataclasses import PseudoSchema
from schema.schema_pseudo_labels import SchemaPseudoLabels
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.zs_tod_turn import TodTurnApiCallCsvRow
import copy

from dataclasses import dataclass


@dataclass
class SchemaWithNameMap:
    schema: PseudoSchema
    name_map: dict[str, str]


class PseudoLabelAugmentation:
    def __init__(self, cfg, num_augmentations=1, api_in_context=False, schemas=None):
        self.cfg = cfg
        self.num_augmentations = num_augmentations
        self.api_in_context = api_in_context
        self.schemas = schemas
        self.pseudo_schemas_map = self.get_pseudo_schemas_and_name_maps()

    def get_pseudo_schemas_and_name_maps(self) -> dict[str, SchemaWithNameMap]:
        spl = SchemaPseudoLabels(self.cfg)
        out = {}
        for i in range(self.num_augmentations):
            pseudo_schema = {}
            pseudo_version_name = self.get_schema_version_name(i)
            for domain, schema in self.schemas.items():
                pseudo_schema, pseudo_name_map = spl.get_pseudo_schema(
                    schema, pseudo_version_name
                )
                out[domain + pseudo_version_name] = SchemaWithNameMap(
                    schema=pseudo_schema,
                    name_map=pseudo_name_map,
                )
        return out

    def apply(self, api_turn: NlgTodTurn, tod_turn: NlgTodTurn):
        aug_data = []
        for i in range(self.num_augmentations):
            aug_turn = copy.deepcopy(api_turn)
            turn_pseudo_schema_maps = [
                self.pseudo_schemas_map[dom + self.get_schema_version_name(i)]
                for dom in api_turn.domains_original
            ]
            self.update_domains_original(aug_turn, turn_pseudo_schema_maps)
            self.update_turn_schemas(aug_turn, turn_pseudo_schema_maps)
            self.update_turn_target(aug_turn, tod_turn, turn_pseudo_schema_maps)
            aug_data.append(aug_turn)
        return aug_data

    def update_domains_original(self, aug_turn: NlgTodTurn, turn_pseudo_name_maps):
        aug_turn.domains_original = [
            schema_with_name_map.schema.service_name
            for schema_with_name_map in turn_pseudo_name_maps
        ]

    def update_turn_schemas(self, aug_turn: NlgTodTurn, turn_pseudo_name_maps):
        aug_turn.schemas = [
            schema_with_name_map.schema
            for schema_with_name_map in turn_pseudo_name_maps
        ]
        pseudo_schema_str = "".join(
            [schema.get_nlg_repr() for schema in aug_turn.schemas]
        )
        aug_turn.schema_str = pseudo_schema_str

    def update_turn_target(self, aug_turn, tod_turn, turn_pseudo_name_maps):
        api_call = copy.deepcopy(tod_turn.context.api_call)
        api_call.method = self.get_pseudo_name_from_maps(
            turn_pseudo_name_maps, api_call.method
        )
        pseudo_params = {
            self.get_pseudo_name_from_maps(
                turn_pseudo_name_maps, param_name
            ): param_value
            for param_name, param_value in api_call.parameters.items()
        }
        api_call.params = pseudo_params
        aug_turn.target.response = str(api_call)
        aug_turn.context.next_system_utterance = aug_turn.target.response

    def get_pseudo_name_from_maps(self, schema_with_name_maps, name):
        for schema_with_name_map in schema_with_name_maps:
            if name in schema_with_name_map.name_map:
                return schema_with_name_map.name_map[name].pseudo_name
        return None

    def get_schema_version_name(self, version):
        return f"pl{version}"


if __name__ == "__main__":
    aiha = PseudoLabelAugmentation(
        DotMap(
            raw_data_root="data/dstc8-schema-guided-dialogue/",
            dataset_name="sgd",
            project_root="/mounts/u-amo-d1/adibm-data/projects/ZSToD",
            prompt_type="default",
            context_type="pseudo_labels",
        )
    )
    aiha.apply()
