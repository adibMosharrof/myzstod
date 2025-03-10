import os
from pathlib import Path
import sys
from typing import Union
import hydra
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from tqdm import tqdm


sys.path.insert(0, os.path.abspath("./src"))
sys.path.insert(0, os.path.abspath("./"))
from generation.generation_handler_factory import GenerationHandlerFactory
from my_enums import Steps
from interpret.activation_statistics import ActivationStatistics
from interpret.activation_masking import ActivationMasks
from interpret.interpret_activations import InterpretActivation

from interpret.activation_plots import ActivationPlots
from metric_managers.metric_manager_factory import MetricManagerFactory

from datamodules.tod_datamodulev2 import TodDataModuleV2
from my_trainers.base_trainer import BaseTrainer
from torch.utils.data import random_split, Subset
from interpret.ordered_activations import OrderedActivations

from datamodules.tod_dataset import TodDataSet
from transformers import AutoConfig
from torch.utils.data import DataLoader, DistributedSampler


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
            interpret_text=interpret_text,
        )

    def get_datasets_for_interpret(self, test_dataset, interpret_text):
        filtered_indices = []
        for i, data in enumerate(test_dataset.data):
            if interpret_text in data.target:
                filtered_indices.append(i)
        filtered_ds = self.get_subset_dataset(
            test_dataset, filtered_indices, Steps.TEST.value, interpret_text
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

        model_config = AutoConfig.from_pretrained(model_out_dir)
        model_config.m_layer = int(self.cfg.interpret_layer)
        model = model_loader.load_for_inference(model_out_dir, config=model_config)

        collate_fn = collator.tod_test_collate
        generation_handler = GenerationHandlerFactory.get_handler(
            self.cfg, model, tokenizer
        )
        metric_manager = MetricManagerFactory.get_metric_manager(
            self.cfg.model_type.context_type, tokenizer, self.logger, self.cfg
        )
        am = ActivationMasks(self.cfg)
        aa = InterpretActivation(self.cfg)
        # aa = OrderedActivations(self.cfg)
        ap = ActivationPlots(self.cfg, am)
        act_stats = ActivationStatistics(self.cfg)

        interpret_texts = [
            # ["ApiCall"]
            ["city", "cuisine", "time", "party_size"],
            ["FindRestaurants", "ReserveRestaurant"],
            # [
            #     "hotel_name",
            #     "check_in_date",
            #     "number_of_days",
            #     "destination",
            #     "city",
            #     "cuisine",
            #     "time",
            #     "party_size",
            # ],
            # ["FindRestaurants", "ReserveRestaurant", "SearchHotel", "ReserveHotel"],
        ]

        for interpret_text in interpret_texts:
            interpret_datasets = [
                self.get_datasets_for_interpret(test_ds, label)
                for label in interpret_text
                for test_ds in test_datasets
            ]
            # generated_tokens, confidence_scores, activations = (
            activations = self.get_generation_with_insights(
                accelerator,
                tokenizer,
                interpret_datasets,
                collate_fn,
                generation_handler,
                metric_manager,
                aa,
            )
            if accelerator.is_main_process:
                activations = [a.cpu().numpy() for a in activations]
                # activations = activations[0]
                # interpret_text = [f"slot_{a}" for a in range(len(activations))]
                ap.plot_activations(activations, interpret_text)
                ap.plot_overlapping_neurons(activations, interpret_text)
                act_stats.activation_statistics(activations, interpret_text)

    def get_generation_with_insights(
        self,
        accelerator,
        tokenizer,
        interpretation_datasets: list[TodDataSet],
        collate_fn,
        generation_handler,
        metric_manager,
        aa: Union[InterpretActivation, OrderedActivations],
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
        for test_dataset in interpretation_datasets:
            dataset_activations = []
            dataset_confidences = []
            test_dl = self.get_test_dl(accelerator, collate_fn, test_dataset)
            for batch in tqdm(test_dl, desc="inference"):
                max_gen_len = self.cfg.max_token_len
                generated_tokens, gen_confidences, fc_vals = (
                    generation_handler.get_generation(
                        batch,
                        max_gen_len - self.cfg.test_prompt_max_len,
                        max_gen_len,
                        self.cfg.test_prompt_max_len,
                        self.cfg.should_post_process,
                        accelerator,
                        metric_manager,
                    )
                )
                activations, confidences = aa.get_activations_and_confidences(
                    test_dataset.interpret_text,
                    generated_tokens,
                    fc_vals,
                    gen_confidences,
                    tokenizer,
                    accelerator,
                )
                dataset_activations.append(activations)
                dataset_confidences.append(confidences)
            # dataset_activations = torch.cat(dataset_activations, dim=0)
            # dataset_confidences = torch.cat(dataset_confidences, dim=0)
            dataset_activations = aa.concat_all_batch_activations(dataset_activations)

            all_activations.append(dataset_activations)
            # all_confidences.append(dataset_confidences)

        accelerator.wait_for_everyone()
        return all_activations


@hydra.main(config_path="../../config/interpret/", config_name="interpret_trainer")
def hydra_start(cfg: DictConfig) -> None:
    torch.manual_seed(42)
    itrainer = InterpretTrainer(cfg)
    itrainer.run()


if __name__ == "__main__":
    hydra_start()
