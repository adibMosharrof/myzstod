from pathlib import Path
from interpret.plots.activation_plotter import ActivationPlotter
from utilities import text_utilities


class AllActivationPlots(ActivationPlotter):

    def plot_activations(self, activations, labels, interpret_datasets):
        # Call the common plot function with default parameters
        labels_text = "_".join(labels)
        if len(labels_text) > 150:
            labels_text = text_utilities.hash_file_name(labels_text, 150, 10)
        subdir_name = Path("all_neurons") / labels_text
        self._plot_activations(
            activations,
            labels,
            save_filename="activation_plots_all",
            neurons_per_plot=12,
            subdir_name=subdir_name,
        )
