import os
from pathlib import Path
import sys
from typing import Union
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig
import pandas as pd
import torch
from tqdm import tqdm


sys.path.insert(0, os.path.abspath("./src"))
sys.path.insert(0, os.path.abspath("./"))
from generation.generation_handler_factory import GenerationHandlerFactory
from my_enums import InterpretFeatureTypes, Steps
from interpret.activation_statistics import ActivationStatistics

from interpret.interpret_utilities import InterpretUtilities
from interpret.interventions.simple_intervention import SimpleIntervention
from interpret.masking.mask_model_base import MaskModelBase
from interpret.masking.max_mask import MaxMask
from interpret.plots.domain_wise_all_activation_plots import (
    DomainWiseAllActivationPlots,
)
from interpret.plots.base_activation_plotter import BaseActivationPlotter
from interpret.activations.interpret_activations import InterpretActivation
from interpret.interpret_features import FeatureInfo, InterpretFeatureGroup
from interpret.plots.overlapping_activation_plots import OverlappingActivationPlots
from interpret.plots.all_activation_plots import AllActivationPlots

from interpret.plots.selective_activation_plots import SelectiveActivationPlots
from metric_managers.metric_manager_factory import MetricManagerFactory

from datamodules.tod_datamodulev2 import TodDataModuleV2
from my_trainers.base_trainer import BaseTrainer
from torch.utils.data import random_split, Subset

from interpret.activations.base_activations import BaseActivations
from datamodules.tod_dataset import TodDataSet
from transformers import AutoConfig
from torch.utils.data import DataLoader, DistributedSampler
from utilities import tensor_utilities


class InterpretTrainer(BaseTrainer):
    def __init__(self, cfg: dict):
        super().__init__(cfg, dm_class=TodDataModuleV2)

    def get_datasets_from_data_modules(self, dms):
        ds = self.get_dm_dataset(dms[0])
        train = ds["train"]
        dataset_size = len(train)
        train_size = int(0.1 * dataset_size)
        dev_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - dev_size
        i_train, i_dev, i_test = random_split(
            range(dataset_size), [train_size, dev_size, test_size]
        )
        train_dataset = self.get_subset_dataset(
            train, i_train.indices, Steps.TRAIN.value
        )
        dev_dataset = self.get_subset_dataset(train, i_dev.indices, Steps.DEV.value)
        test_dataset = self.get_subset_dataset(train, i_test.indices, Steps.TEST.value)
        return train_dataset, dev_dataset, [test_dataset]

    def get_subset_dataset(self, dataset, indices, step, interpret_text=""):
        return TodDataSet(
            [dataset.data[i] for i in indices],
            dataset.dataset_name,
            dataset.domain_setting,
            step_name=step,
            raw_data_root=dataset.raw_data_root,
            interpret_feature_info=interpret_text,
        )

    def get_datasets_for_interpret(self, test_dataset, feature_info: FeatureInfo):
        filtered_indices = []
        for i, data in enumerate(test_dataset.data):
            if feature_info.name in data.target:
                filtered_indices.append(i)
        filtered_ds = self.get_subset_dataset(
            test_dataset, filtered_indices, Steps.TEST.value, feature_info
        )
        return filtered_ds

    def inference(
        self,
        accelerator,
        tokenizer,
        model_loader,
        collator,
        test_datasets,
        model_out_dir,
        train_dataset,
    ):
        if isinstance(self.cfg.interpret_layer, int):
            return self.inference_for_layer(
                accelerator,
                tokenizer,
                model_loader,
                collator,
                test_datasets,
                model_out_dir,
                train_dataset,
                self.cfg.interpret_layer,
            )
        if isinstance(self.cfg.interpret_layer, ListConfig):
            [
                self.inference_for_layer(
                    accelerator,
                    tokenizer,
                    model_loader,
                    collator,
                    test_datasets,
                    model_out_dir,
                    train_dataset,
                    i,
                )
                for i in self.cfg.interpret_layer
            ]

    def inference_for_layer(
        self,
        accelerator,
        tokenizer,
        model_loader,
        collator,
        test_datasets,
        model_out_dir,
        train_dataset,
        interpret_layer,
    ):
        model_config = AutoConfig.from_pretrained(model_out_dir)
        # model_config.m_layer = int(self.cfg.interpret_layer)
        model_config.m_layer = interpret_layer
        model = model_loader.load_for_inference(model_out_dir, config=model_config)

        collate_fn = collator.tod_test_collate
        generation_handler = GenerationHandlerFactory.get_handler(
            self.cfg, model, tokenizer
        )
        metric_manager = MetricManagerFactory.get_metric_manager(
            self.cfg.model_type.context_type, tokenizer, self.logger, self.cfg
        )
        # am = ActivationMasks(self.cfg)
        am = MaxMask(self.cfg)
        aa = InterpretActivation(self.cfg)
        # aa = OrderedActivations(self.cfg)
        act_stats = ActivationStatistics(self.cfg, interpret_layer)
        oap = OverlappingActivationPlots(self.cfg, am, interpret_layer)
        activation_plots: list[BaseActivationPlotter] = [
            # DomainWiseAllActivationPlots(self.cfg, am, interpret_layer, act_stats, oap),
            AllActivationPlots(self.cfg, am, interpret_layer),
            SelectiveActivationPlots(self.cfg, am, interpret_layer),
        ]

        mask_model_type = MaskModelBase()
        interventions = [
            SimpleIntervention(
                self.cfg,
                mask=am,
                mask_percent=self.cfg.percent_mask,
                mask_model_type=mask_model_type,
            ),
        ]

        all_interpret_features = self.get_feature_data()
        for interpret_features in tqdm(all_interpret_features, desc="all interpret"):
            interpret_datasets = [
                self.get_datasets_for_interpret(test_ds, feature)
                for feature in interpret_features.features
                for test_ds in test_datasets
            ]
            # generated_tokens, confidence_scores, activations = (
            activations = self.get_generation_with_insights(
                accelerator,
                tokenizer,
                interpret_datasets,
                collate_fn,
                generation_handler,
                aa,
                interpret_features,
                intervention_group_name=interpret_features.group_name,
                intervention_name="base",
                interpret_layer=interpret_layer,
            )
            used_datasets = InterpretUtilities.get_used_datasets(interpret_datasets)
            for index, interpret_dataset in enumerate(used_datasets):
                for intervention in interventions:
                    generation_handler.model = intervention.intervene(
                        model, activations, used_datasets, index
                    )
                    feature_name, feature_domain = (
                        interpret_dataset.interpret_feature_info.name,
                        interpret_dataset.interpret_feature_info.domain,
                    )

                    intervention_name = intervention.get_intervention_name(
                        feature_name, feature_domain
                    )
                    self.get_generation_with_insights(
                        accelerator,
                        tokenizer,
                        used_datasets,
                        collate_fn,
                        generation_handler,
                        aa,
                        interpret_features,
                        intervention_group_name=interpret_features.group_name,
                        intervention_name=intervention_name,
                        interpret_layer=interpret_layer,
                    )

            if accelerator.is_main_process:
                activations = [a.cpu().numpy() for a in activations]
                # activations = activations[0]
                # interpret_text = [f"slot_{a}" for a in range(len(activations))]
                [
                    ap.plot_activations(
                        activations,
                        interpret_features.get_feature_names(),
                        interpret_datasets,
                    )
                    for ap in activation_plots
                ]
                # act_stats.activation_statistics(
                #     activations,
                #     interpret_features.get_feature_names(),
                #     interpret_datasets,
                # )

    def get_generation_with_insights(
        self,
        accelerator,
        tokenizer,
        interpretation_datasets: list[TodDataSet],
        collate_fn,
        generation_handler,
        aa: BaseActivations,
        interpret_features: InterpretFeatureGroup,
        intervention_group_name="default",
        intervention_name="base",
        interpret_layer=11,
    ):
        p = (
            self.cfg.project_root
            / "data_exploration"
            / "interpret_activations"
            / "activations.pt"
        )
        if p.exists() and 0:
            all_activations = torch.load(p)
            return all_activations
        all_activations = []
        all_confidences = []
        metric_manager = MetricManagerFactory.get_metric_manager(
            self.cfg.model_type.context_type, tokenizer, self.logger, self.cfg
        )
        for test_dataset in tqdm(interpretation_datasets, desc="features"):
            if not test_dataset.data:
                continue
            feature_types = interpret_features.get_feature_types()
            metric_manager.initialize(
                feature_types,
                test_dataset.interpret_feature_info.name,
                intervention_group_name,
                intervention_name,
                interpret_layer,
            )
            dataset_activations = []
            dataset_confidences = []
            dataset_gen_tokens = []
            dataset_labels = []
            test_dl = self.get_test_dl(accelerator, collate_fn, test_dataset)
            for batch in tqdm(test_dl, desc="inference"):
                max_gen_len = self.cfg.max_token_len
                context_len = self.cfg.test_prompt_max_len
                generated_tokens, gen_confidences, fc_vals = (
                    generation_handler.get_generation(
                        batch,
                        max_gen_len - context_len,
                        max_gen_len,
                        context_len,
                        self.cfg.should_post_process,
                        accelerator,
                        metric_manager,
                    )
                )
                dataset_gen_tokens.append(generated_tokens)
                dataset_labels.append(batch.labels)
                activations, confidences = aa.get_activations_and_confidences(
                    test_dataset.interpret_feature_info.name,
                    generated_tokens,
                    fc_vals,
                    gen_confidences,
                    tokenizer,
                    accelerator,
                )
                dataset_activations.append(activations)
                dataset_confidences.append(gen_confidences)

            dataset_gen_tokens = torch.cat(
                tensor_utilities.pad_tensor_batch(
                    dataset_gen_tokens, pad_value=tokenizer.pad_token_id
                ),
                dim=0,
            )
            dataset_labels = torch.cat(
                tensor_utilities.pad_tensor_batch(
                    dataset_labels, pad_value=tokenizer.pad_token_id
                ),
                dim=0,
            )
            metric_manager.compute_metrics(dataset_gen_tokens, dataset_labels)
            # dataset_activations = torch.cat(dataset_activations, dim=0)
            # dataset_confidences = torch.cat(dataset_confidences, dim=0)
            dataset_activations = aa.concat_all_batch_activations(dataset_activations)

            all_activations.append(dataset_activations)
            # all_confidences.append(dataset_confidences)

        accelerator.wait_for_everyone()
        torch.save(all_activations, p)
        return all_activations

    def get_feature_data(self):

        feature_data = {
            "restaurant": {
                InterpretFeatureTypes.PARAM: ["city", "cuisine", "time", "party_size"],
                InterpretFeatureTypes.METHOD: ["FindRestaurants", "ReserveRestaurant"],
            },
            # # Add more domains dynamically
            # "media": {
            #     InterpretFeatureTypes.PARAM: [
            #         "title",
            #         "subtitles",
            #         # "check_out_date",
            #         # "room_type",
            #     ],
            #     InterpretFeatureTypes.METHOD: ["FindMovies", "PlayMovie"],
            # },
            "hotels": {
                InterpretFeatureTypes.PARAM: [
                    "hotel_name",
                    "check_in_date",
                    "number_of_days",
                    "destination",
                    "number_of_rooms",
                ],
                InterpretFeatureTypes.METHOD: ["ReserveHotel", "SearchHotel"],
            },
            "banks": {
                InterpretFeatureTypes.PARAM: [
                    "account_type",
                    "transaction_type",
                    "amount",
                    "recipient_account_name",
                ],
                InterpretFeatureTypes.METHOD: [
                    "CheckBalance",
                    "TransferMoney",
                ],
            },
            # "movies": {
            #     InterpretFeatureTypes.PARAM: [
            #         "location",
            #         "movie_name",
            #         "number_of_tickets",
            #         "show_date",
            #         "show_time",
            #         "show_type",
            #         "genre",
            #     ],
            #     InterpretFeatureTypes.METHOD: ["BuyMovieTickets", "FindMovies"],
            # },
        }
        feature_data = {
            "restaurant": {
                InterpretFeatureTypes.PARAM: ["city"],
                # InterpretFeatureTypes.METHOD: ["FindRestaurants", "ReserveRestaurant"],
            },
            "hotels": {
                InterpretFeatureTypes.PARAM: [
                    "destination",
                ],
                # InterpretFeatureTypes.METHOD: ["ReserveHotel", "SearchHotel"],
            },
        }

        # Generate feature groups dynamically for params and methods separately
        param_features = InterpretFeatureGroup(
            [
                FeatureInfo(name, domain, InterpretFeatureTypes.PARAM.value)
                for domain, details in feature_data.items()
                for name in details.get(InterpretFeatureTypes.PARAM, [])
            ],
            "params",
        )

        method_features = InterpretFeatureGroup(
            [
                FeatureInfo(name, domain, InterpretFeatureTypes.METHOD.value)
                for domain, details in feature_data.items()
                for name in details.get(InterpretFeatureTypes.METHOD, [])
            ],
            "methods",
        )

        # Generate combined feature groups (params + methods) for each domain
        # domain_feature_groups = {
        # domain: InterpretFeatureGroup(
        domain_feature_groups = InterpretFeatureGroup(
            np.concatenate(
                [
                    [
                        FeatureInfo(name, domain, InterpretFeatureTypes.PARAM.value)
                        for name in details.get(InterpretFeatureTypes.PARAM, [])
                    ]
                    + [
                        FeatureInfo(name, domain, InterpretFeatureTypes.METHOD.value)
                        for name in details.get(InterpretFeatureTypes.METHOD, [])
                    ]
                    for domain, details in feature_data.items()
                ]
            ),
            "domains",
        )
        # }

        # all_interpret_features = [
        #     # method_features,
        #     # domain_feature_groups,
        #     param_features,
        # ]
        all_interpret_features = [
            # method_features,
            # domain_feature_groups,
            param_features,
        ]
        return all_interpret_features


@hydra.main(config_path="../../config/interpret/", config_name="interpret_trainer")
def hydra_start(cfg: DictConfig) -> None:
    torch.manual_seed(42)
    itrainer = InterpretTrainer(cfg)
    itrainer.run()


if __name__ == "__main__":
    hydra_start()
