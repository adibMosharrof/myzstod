from abc import ABC, abstractmethod
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch

from interpret.masking.base_mask import BaseMask


class BaseActivationPlotter(ABC):
    def __init__(self, cfg, activation_masks: BaseMask, interpret_layer):
        self.cfg = cfg
        self.interpret_layer = interpret_layer
        self.activation_masks = activation_masks

    def mad(self, activations):
        median = np.median(activations, axis=0)
        deviations = np.abs(activations - median)
        mad = np.median(deviations, axis=0)
        return mad

    def convert_activations(self, activations):
        """Converts torch tensor activations to numpy arrays."""
        if isinstance(activations, torch.Tensor):
            return activations.cpu().numpy()
        return activations

    def get_sorted_neurons(self, activations):
        """Returns neurons sorted by their mean absolute activation values."""
        fc_class1 = np.array(activations[0])
        mean_class0 = np.mean(np.abs(fc_class1), axis=0)
        return np.argsort(mean_class0)

    def save_plot(self, file_name, subdir):
        """Saves the plot to the specified directory."""
        out_path = (
            Path("interpret_activations")
            / subdir
            / f"layer_{self.interpret_layer}"
            / file_name
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), format="pdf", bbox_inches="tight", dpi=600)
        plt.close()

    def normalize_activations(self, activations):
        """Normalizes activations to the range [0, 1]."""
        return (activations - np.mean(activations)) / np.std(activations)

    def plot_activations(self, activations, labels):
        raise NotImplementedError()
