from pathlib import Path
import numpy as np
import pandas as pd
import torch
from interpret.plots.base_activation_plotter import BaseActivationPlotter
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os


class OverlappingActivationPlots(BaseActivationPlotter):
    def plot_activations(
        self,
        activations,
        labels,
        interpret_datasets,
        max_neurons_to_plot=10,
        neurons_per_figure=10,
        subdir_name="overlapping_neurons",
    ):
        all_masked_indices = [
            np.where(self.activation_masks.get_mask(act, self.cfg.percent_mask) == 0)[0]
            for act in activations
        ]
        common_neurons = self.find_common_neurons(all_masked_indices)
        if not common_neurons:
            return
        # neurons_to_plot = np.random.choice(
        #     common_neurons,
        #     size=min(len(common_neurons), max_neurons_to_plot),
        #     replace=False,
        # )
        # plot_df = self.prepare_neuron_data(activations, neurons_to_plot, labels)
        # self._create_activation_plot(plot_df, neurons_to_plot, labels)
        neuron_chunks = [
            common_neurons[i : i + neurons_per_figure]
            for i in range(0, len(common_neurons), neurons_per_figure)
        ]

        with Pool() as pool:
            pool.starmap(
                self._plot_neuron_chunk,
                [
                    (chunk, activations, labels, idx, subdir_name)
                    for idx, chunk in enumerate(neuron_chunks)
                ],
            )

    def _plot_neuron_chunk(
        self, neurons_to_plot, activations, labels, chunk_idx, subdir_name
    ):
        plot_df = self.prepare_neuron_data(activations, neurons_to_plot, labels)
        self._create_activation_plot(
            plot_df, neurons_to_plot, labels, chunk_idx, subdir_name
        )

    def prepare_neuron_data(self, activations, neurons, labels):
        plot_data = []
        masks = [self.activation_masks.get_mask(act, 0.50) for act in activations]
        for class_label, activation, mask in zip(labels, activations, masks):
            class_neuron_indices = np.where(mask == 0)[0]
            common_neuron_indices = class_neuron_indices[
                np.isin(class_neuron_indices, neurons)
            ]
            normalized_activations = self.normalize_activations(
                activation[:, common_neuron_indices]
            )
            df = pd.DataFrame(normalized_activations).melt(
                var_name="Neuron_Local", value_name="Activation"
            )
            df["Neuron"] = df["Neuron_Local"].apply(lambda x: common_neuron_indices[x])
            df["Class"] = class_label
            plot_data.append(df)
        return pd.concat(plot_data, ignore_index=True)

    def find_common_neurons(self, all_masked_indices):
        if not len(all_masked_indices):
            return []
        common_neurons = set(all_masked_indices[0])
        for indices in all_masked_indices[1:]:
            common_neurons.intersection_update(indices)
        return list(common_neurons)

    def _create_activation_plot(
        self, plot_df, neurons_to_plot, labels, figure_idx, subdir_name
    ):
        plt.figure(figsize=(15, 8))
        palette = sns.color_palette("husl", n_colors=len(labels))
        sns.boxplot(
            data=plot_df,
            x="Neuron",
            y="Activation",
            hue="Class",
            showfliers=False,
            palette=palette,
            width=0.5,
            dodge=True,
            linewidth=0.3,
            gap=0.1,
        )
        for pos in range(len(neurons_to_plot)):
            plt.axvline(x=pos + 0.5, color="black", linestyle="--", linewidth=1)
        plt.xticks(np.arange(len(neurons_to_plot)), neurons_to_plot, fontsize=10)
        plt.title(
            f"Activation Box Plot for {len(neurons_to_plot)} Common Neurons",
            fontsize=18,
        )
        plt.xlabel("Neuron Index", fontsize=14)
        plt.ylabel("Activation Value", fontsize=14)
        plt.legend(
            title="Class", fontsize=12, bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.tight_layout()
        folder_name = Path(subdir_name) / "overlapping_neurons"
        file_name = f"overlapping_neurons_{figure_idx}"
        self.save_plot(file_name + ".pdf", folder_name)
        plt.close()
