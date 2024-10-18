from dotmap import DotMap

from datamodules.data_collators.base_collator import BaseCollator
from my_enums import ZsTodConstants
from schema.pseudo_schema_dataclasses import PseudoSchema
from schema.schema_pseudo_labels import SchemaPseudoLabels
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.api_call_turn_csv_row import ApiCallTurnCsvRow
from tod.turns.zs_tod_turn import TodTurnApiCallCsvRow
import copy

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SchemaWithNameMap:
    schema: PseudoSchema
    name_map: dict[str, str]


class PseudoLabelAugmentation:
    def __init__(
        self,
        cfg,
        num_augmentations=1,
        api_in_context=False,
        schemas=None,
        collator: BaseCollator = None,
        turn_row_csv_cls=ApiCallTurnCsvRow,
    ):
        self.cfg = cfg
        self.num_augmentations = num_augmentations
        self.api_in_context = api_in_context
        self.schemas = schemas
        self.pseudo_schemas_map = self.get_pseudo_schemas_and_name_maps()
        self.collator = collator
        self.turn_row_csv_cls = turn_row_csv_cls

    def get_pseudo_schemas_and_name_maps(self) -> dict[str, SchemaWithNameMap]:
        spl = SchemaPseudoLabels(self.cfg)
        out = {}
        pseudo_schemas = []
        for i in range(self.num_augmentations):
            pseudo_version_name = self.get_schema_version_name(i)
            for domain, schema in self.schemas.items():
                pseudo_schema, pseudo_name_map = spl.get_pseudo_schema(
                    schema, pseudo_version_name
                )
                out[domain + pseudo_version_name] = SchemaWithNameMap(
                    schema=pseudo_schema,
                    name_map=pseudo_name_map,
                )
                pseudo_schemas.append(pseudo_schema)
        for pseudo_schema in pseudo_schemas:
            self.schemas[pseudo_schema.service_name] = pseudo_schema
        return out

    def apply(self, api_turn: NlgTodTurn = None, tod_turn: NlgTodTurn = None):
        aug_turns = []
        if api_turn:
            aug_turns.extend(self.apply_api_call_augmentation(api_turn, tod_turn))
        # api_turn_augs = self.apply_api_in_context_augmentation(api_turn, tod_turn)
        # aug_turns.extend(api_turn_augs)
        return aug_turns

    def apply_api_in_context_augmentation(
        self, api_turn: NlgTodTurn = None, tod_turn: NlgTodTurn = None
    ):
        csv_item = self.turn_row_csv_cls.to_csv_row(
            context_type=self.cfg.context_type, tod_turn=api_turn
        )
        turn_csv_row_item = TodTurnApiCallCsvRow.from_list_of_values_and_headers(
            values=csv_item, headers=self.turn_row_csv_cls.get_csv_headers()
        )
        context_text = self.collator.get_context_text(turn_csv_row_item)
        api_call_tokens = self.collator.tokenizer.encode(ZsTodConstants.API_CALL.value)
        context_tokens, unused_len = self.collator.get_context_tokens_and_unused_len(
            turn_csv_row_item, context_text
        )
        is_api_in_context = self.is_contiguous_subsequence(
            api_call_tokens, context_tokens
        )
        pass

    def apply_api_call_augmentation(self, api_turn: NlgTodTurn, tod_turn: NlgTodTurn):
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
            aug_turn.dataset_name = (
                self.cfg.dataset_name + self.get_schema_version_name(i)
            )
            aug_data.append(aug_turn)
        return aug_data

    def is_contiguous_subsequence(self, tensor1, tensor2):
        # Reshape tensors for conv1d (batch_size, channels, length)
        tensor1 = torch.tensor(tensor1).view(1, 1, -1).float()
        tensor2 = torch.tensor(tensor2).view(1, 1, -1).float()

        # Convolution with tensor1 as the kernel
        conv_result = F.conv1d(tensor2, tensor1)

        # Check if any convolution result matches the exact length of tensor1
        return torch.any(conv_result == tensor1.size(-1))

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
