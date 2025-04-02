from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis, normaltest, shapiro, kstest, norm


class ActivationStatistics:

    def __init__(self, cfg, interpret_layer):
        self.cfg = cfg
        self.interpret_layer = interpret_layer

    def activation_statistics(self, activations, interpret_text, base_path):
        for activation, label in zip(activations, interpret_text):
            n_neurons = activation.shape[1]
            total_skew = 0
            total_kurtosis = 0
            # Initialize statistics
            total_skew = 0
            total_kurtosis = 0
            shapiro_pvals = []
            normal_pvals = []

            # Test each neuron
            for neuron in range(n_neurons):
                neuron_data = activation[:, neuron]
                data_mean = np.mean(neuron_data)
                data_std = np.std(neuron_data)

                neuron_data = (neuron_data - data_mean) / data_std
                # Basic statistics
                total_skew += skew(neuron_data)
                total_kurtosis += kurtosis(
                    neuron_data, fisher=False
                )  # fisher=False for Pearson's definition

                # Statistical tests
                try:
                    _, shapiro_p = shapiro(neuron_data)
                except ValueError:
                    shapiro_p = -1
                try:
                    _, normal_p = normaltest(
                        neuron_data
                    )  # D'Agostino and Pearson's test
                except ValueError:
                    normal_p = -1

                shapiro_pvals.append(shapiro_p)
                normal_pvals.append(normal_p)
                ks_stat, ks_p_value = kstest(
                    neuron_data,
                    "norm",
                    args=(data_mean, data_std),
                )

            # Average statistics
            avg_skew = total_skew / n_neurons
            avg_kurtosis = total_kurtosis / n_neurons

            # Count how many neurons pass normality tests (p > 0.05)
            shapiro_normal = sum(p > 0.05 for p in shapiro_pvals)
            dagostino_normal = sum(p > 0.05 for p in normal_pvals)
            file_path = (
                base_path / Path("statistics") / f"activation_statistics_{label}.txt"
            )
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                f.write(f"Label: {label}\n")
                f.write(f"Average Skewness: {avg_skew:.4f}\n")
                f.write(f"Average Kurtosis: {avg_kurtosis:.4f}\n")
                f.write(f"Neurons passing Shapiro-Wilk: {shapiro_normal}/{n_neurons}\n")
                f.write(
                    f"Neurons passing D'Agostino-Pearson: {dagostino_normal}/{n_neurons}\n"
                )
                f.write(f"KS Test Statistic: {ks_stat:.4f}, p-value: {ks_p_value:.4f}")
                f.write(
                    f"  Interpretation: {'Likely Normal' if ks_p_value > 0.05 else 'Not Normal'}\n"
                )
