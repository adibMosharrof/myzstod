from pathlib import Path
import numpy as np
from interpret.plots.activation_plotter import ActivationPlotter
from utilities import text_utilities


class SelectiveActivationPlots(ActivationPlotter):

    def plot_activations(self, activations, labels, interpret_datasets):
        # Select specific neurons (top, center, and bottom neurons)
        activations = self.convert_activations(activations)
        if len(activations) <= 1:
            return

        sorted_neurons = self.get_sorted_neurons(activations)
        selected_neurons = np.concatenate(
            [
                sorted_neurons[-2:],  # Top 2
                sorted_neurons[
                    len(sorted_neurons) // 2 : len(sorted_neurons) // 2 + 2
                ],  # Center 2
                sorted_neurons[:2],  # Bottom 2
            ]
        )

        labels_text = "_".join(labels)
        if len(labels_text) > 150:
            labels_text = text_utilities.hash_file_name(labels_text, 150, 10)
        subdir_name = Path("selected_neurons") / labels_text
        # Call the common plot function with specific parameters
        self._plot_activations(
            activations,
            labels,
            selected_neurons=selected_neurons,
            save_filename="activation_plots_selected",
            neurons_per_plot=6,
            subdir_name=subdir_name,
        )
