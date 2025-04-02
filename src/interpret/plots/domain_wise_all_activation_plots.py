from pathlib import Path

import numpy as np
from interpret.activation_statistics import ActivationStatistics
from interpret.interpret_utilities import InterpretUtilities
from interpret.masking.base_mask import BaseMask
from interpret.plots.activation_plotter import ActivationPlotter
from interpret.plots.overlapping_activation_plots import OverlappingActivationPlots
from my_enums import InterpretFeatureTypes


class DomainWiseAllActivationPlots(ActivationPlotter):
    def __init__(
        self,
        cfg,
        activation_masks: BaseMask,
        interpret_layer,
        activation_stats: ActivationStatistics,
        overlapping_activation_plots: OverlappingActivationPlots,
    ):
        super().__init__(cfg, activation_masks, interpret_layer)
        self.activation_stats = activation_stats
        self.overlapping_activation_plots = overlapping_activation_plots

    def plot_activations(self, activations, labels, interpret_datasets):
        # Call the common plot function with default parameters

        subdir_name = Path("domain_wise_neurons")
        # used_datasets = [d for d in interpret_datasets if d.data]
        used_datasets = InterpretUtilities.get_used_datasets(interpret_datasets)
        grouped_activations = {}
        for i, dataset in enumerate(used_datasets):
            domain = dataset.interpret_feature_info.domain
            if domain not in grouped_activations:
                grouped_activations[domain] = activations[i]
            else:

                grouped_activations[domain] = np.concatenate(
                    [grouped_activations[domain], activations[i]]
                )
        group_act_values = list(grouped_activations.values())
        group_act_keys = list(grouped_activations.keys())
        subdir_path = subdir_name / "all"
        self._plot_activations(
            group_act_values,
            group_act_keys,
            save_filename="domain_wise",
            neurons_per_plot=12,
            subdir_name=subdir_path,
        )
        self.overlapping_activation_plots.plot_activations(
            group_act_values,
            group_act_keys,
            interpret_datasets,
            subdir_name=subdir_path,
        )
        self.activation_stats.activation_statistics(
            group_act_values,
            group_act_keys,
            "interpret_activations" / subdir_path / f"layer_{self.interpret_layer}",
        )

        for feature_type in InterpretFeatureTypes.list():
            grouped_activations = {}
            for i, dataset in enumerate(used_datasets):
                if dataset.interpret_feature_info.feature_type != feature_type:
                    continue
                key = dataset.interpret_feature_info.domain + "_" + feature_type
                if domain not in grouped_activations:
                    grouped_activations[key] = activations[i]
                else:
                    grouped_activations[key] = np.concatenate(
                        [grouped_activations[key], activations[i]]
                    )
            group_act_values = list(grouped_activations.values())
            group_act_keys = list(grouped_activations.keys())
            subdir_path = subdir_name / feature_type
            self._plot_activations(
                group_act_values,
                group_act_keys,
                save_filename=f"domain_wise_{feature_type}",
                neurons_per_plot=12,
                subdir_name=subdir_path,
            )
            self.activation_stats.activation_statistics(
                group_act_values,
                group_act_keys,
                "interpret_activations" / subdir_path / f"layer_{self.interpret_layer}",
            )
            self.overlapping_activation_plots.plot_activations(
                group_act_values,
                group_act_keys,
                interpret_datasets,
                subdir_name=subdir_path,
            )
