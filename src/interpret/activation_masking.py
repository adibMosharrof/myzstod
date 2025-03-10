import numpy as np
from scipy import stats


class ActivationMasks:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_mask_max(self, activations, percent):
        mean_vals = np.mean(np.abs(activations), axis=0)
        sorted_indices = np.argsort(mean_vals)[::-1]
        mask_count = int(percent * len(mean_vals))
        mask = np.ones_like(mean_vals)
        mask[sorted_indices[:mask_count]] = 0
        return mask

    def get_kst_mask(self, activations, percent):
        n_neurons = activations.shape[1]

        kst_score = np.zeros(n_neurons)
        for neuron in range(n_neurons):
            neuron_data = activations[:, neuron]
            # Normalize data
            normalized_data = (neuron_data - np.mean(neuron_data)) / np.std(neuron_data)
            # Run KS test against standard normal
            ks_stat, _ = stats.kstest(normalized_data, "norm")

            kst_score[neuron] = ks_stat

        sorted_indices = np.argsort(kst_score)[::-1]

        mask_count = int(percent * n_neurons)
        mask = np.ones_like(kst_score)
        mask[sorted_indices[:mask_count]] = 0.0
        return mask
