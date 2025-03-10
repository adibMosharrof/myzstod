from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

from interpret.activation_masking import ActivationMasks


class ActivationPlots:
    def __init__(self, cfg, activation_masks: ActivationMasks):
        self.cfg = cfg
        self.activation_masks = activation_masks

    def plot_activations(self, activations, labels):
        if not len(activations) > 1:
            return
        # activations = [a.cpu().numpy() for a in activations]
        if type(activations) == torch.Tensor:
            activations = activations.cpu().numpy()
        fc_class1 = np.array(activations[0])
        mean_class0 = np.mean(np.abs(fc_class1), axis=0)
        sorted_neurons = np.argsort(mean_class0)

        top_neurons = sorted_neurons[-2:]
        center_neurons = sorted_neurons[
            int(mean_class0.shape[0] / 2) : int(mean_class0.shape[0] / 2) + 2
        ]
        bottom_neurons = sorted_neurons[:2]  # Bottom 2
        selected_neurons = np.concatenate([top_neurons, center_neurons, bottom_neurons])
        num_neurons = (len(selected_neurons) + 1) // 2
        num_classes = len(activations)
        fig, axs = plt.subplots(num_neurons, 2, figsize=(12, 10))
        axs = axs.ravel()

        # Seaborn color palette
        # colors = sns.color_palette("Set2", n_colors=num_classes)
        colors = sns.color_palette("husl", n_colors=num_classes)

        # Plot KDE distributions for each selected neuron
        for idx, neuron_idx in enumerate(selected_neurons):
            for class_idx, fc_class in enumerate(activations):
                dist_class = [n[neuron_idx] for n in fc_class]
                sns.kdeplot(
                    dist_class,
                    fill=True,
                    color=colors[class_idx],
                    label=labels[class_idx] if idx == 0 else None,
                    ax=axs[idx],
                    alpha=0.6,
                )

                # Set subplot titles
            if idx < 2:
                title = f"Top Neuron {neuron_idx}"
            elif idx < 4:
                title = f"Center Neuron {neuron_idx}"
            else:
                title = f"Bottom Neuron {neuron_idx}"

            axs[idx].set_title(title)
            axs[idx].set_xlabel("Activation Value")
            axs[idx].set_ylabel("Density")

            # Add legend to the first plot only
            if idx == 0:
                axs[idx].legend()

        plt.tight_layout()
        file_name = "activation_plots" + " ".join(labels) + ".pdf"

        out_path = Path("interpret_activations") / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save figure
        plt.savefig(str(out_path), format="pdf", bbox_inches="tight", dpi=600)
        plt.close()

    def mad(self, activations):
        median = np.median(activations, axis=0)
        deviations = np.abs(activations - median)
        mad = np.median(deviations, axis=0)
        return mad

    def plot_overlapping_neurons(self, activations, labels, max_neurons_to_plot=10):
        all_masked_indices = []
        for activation in activations:
            mask_max = self.activation_masks.get_mask_max(
                activation, self.cfg.percent_mask
            )
            class_neuron_indices = np.where(mask_max == 0)[0]
            # class_neuron_indices = [int(neuron) for neuron in class_neuron_indices]
            all_masked_indices.append(class_neuron_indices)
        common_neurons = self.find_common_neurons(all_masked_indices)
        neurons_to_plot = np.random.choice(
            common_neurons,
            size=min(len(common_neurons), max_neurons_to_plot),
            replace=False,
        )
        # neurons_to_plot = [int(neuron) for neuron in neurons_to_plot]
        plot_df = self.prepare_neuron_data(activations, neurons_to_plot, labels)
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
        # neuron_indices = [0, 4, 6, 12]  # Example indices where dashed lines should appear
        for pos in range(len(neurons_to_plot)):  # Use `neurons_to_plot` here
            plt.axvline(x=pos + 0.5, color="black", linestyle="--", linewidth=1)

        plt.xticks(np.arange(0, len(neurons_to_plot), 1), neurons_to_plot, fontsize=10)
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
        file_name = "overlapping_neurons" + " ".join(labels) + ".pdf"
        out_path = Path("interpret_activations") / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), format="pdf", bbox_inches="tight", dpi=600)
        plt.close()

    def prepare_neuron_data(self, activations, neurons, labels):
        plot_data = []
        masks = [
            self.activation_masks.get_mask_max(activation, 0.50)
            for activation in activations
        ]
        for class_label, activation, mask in zip(labels, activations, masks):
            class_neuron_indices = np.where(mask == 0)[0]

            common_neuron_mask = np.isin(class_neuron_indices, neurons)
            common_neuron_mask = np.where(common_neuron_mask, 0, 1)
            common_neuron_indices = class_neuron_indices[common_neuron_mask == 0]
            filtered_activations = activation[:, common_neuron_indices]
            normalized_activations = (
                filtered_activations - np.mean(filtered_activations)
            ) / np.std(filtered_activations)
            df = pd.DataFrame(normalized_activations)
            df = df.melt(var_name="Neuron_Local", value_name="Activation")
            df["Neuron"] = df["Neuron_Local"].apply(lambda x: common_neuron_indices[x])
            df["Class"] = class_label
            plot_data.append(df)
        return pd.concat(plot_data, ignore_index=True)

    def find_common_neurons(self, all_masked_indices):
        masked_indices = [a.tolist() for a in all_masked_indices]
        common_neurons = set(masked_indices[0])
        for indices in masked_indices[1:]:
            common_neurons = common_neurons.intersection(indices)
        return list(common_neurons)
