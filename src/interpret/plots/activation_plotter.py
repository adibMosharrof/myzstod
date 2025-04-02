from pathlib import Path
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool

from tqdm import tqdm
from interpret.plots.base_activation_plotter import BaseActivationPlotter


class ActivationPlotter(BaseActivationPlotter):

    def _plot_single_task(
        self,
        plot_idx,
        activations,
        labels,
        selected_neurons,
        neurons_per_plot,
        num_neurons,
        subdir_name,
        save_file_name,
    ):
        start_idx = plot_idx * neurons_per_plot
        end_idx = min((plot_idx + 1) * neurons_per_plot, num_neurons)

        # Adjust subplot layout to match the number of neurons in this task
        neurons_in_task = end_idx - start_idx
        if neurons_in_task < 9:
            num_rows, num_cols = 3, 3
        else:
            num_rows = (
                neurons_in_task + 3
            ) // 4  # We want a maximum of 4 neurons per row
            num_cols = min(4, neurons_in_task)  # Maximum 4 columns

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows))
        axs = (
            axs.ravel() if neurons_in_task > 1 else [axs]
        )  # Flatten axs if more than 1 subplot, or make it a single item

        colors = sns.color_palette("bright", n_colors=len(activations))

        # Plot activations for each selected neuron
        for idx, neuron_idx in enumerate(selected_neurons[start_idx:end_idx]):
            for class_idx, fc_class in enumerate(activations):
                sns.kdeplot(
                    [n[neuron_idx] for n in fc_class],
                    fill=True,
                    color=colors[class_idx],
                    label=labels[class_idx] if idx == 0 else None,
                    ax=axs[idx],
                    alpha=0.6,
                )
            axs[idx].set_title(f"Neuron {neuron_idx}")
            axs[idx].set_xlabel("Activation Value")
            axs[idx].set_ylabel("Density")
            if idx == 0:
                axs[idx].legend()

        plt.tight_layout()
        neuron_numbers = "_".join(str(n) for n in selected_neurons[start_idx:end_idx])
        file_name = f"{save_file_name}_{neuron_numbers}.pdf"
        self.save_plot(file_name, subdir_name)

    def _plot_activations(
        self,
        activations,
        labels,
        selected_neurons=None,
        neurons_per_plot=6,
        save_filename="activation_plots",
        subdir_name="all_neurons",
    ):
        activations = self.convert_activations(activations)
        if len(activations) <= 1:
            return

        # If no specific neurons are selected, use all neurons
        if selected_neurons is None:
            sorted_neurons = self.get_sorted_neurons(activations)
            selected_neurons = sorted_neurons
        num_neurons = len(selected_neurons)

        num_plots = (num_neurons + neurons_per_plot - 1) // neurons_per_plot

        # Use multiprocessing to parallelize the plotting tasks
        with Pool() as pool:
            tasks = [
                pool.apply_async(
                    self._plot_single_task,
                    args=(
                        plot_idx,
                        activations,
                        labels,
                        selected_neurons,
                        neurons_per_plot,
                        num_neurons,
                        subdir_name,
                        save_filename,
                    ),
                )
                for plot_idx in range(num_plots)
            ]
            # Wait for all tasks to complete
            for task in tqdm(tasks, desc="Plotting activations", total=num_plots):
                task.get()

        print(f"All plots saved to {save_filename}.")
