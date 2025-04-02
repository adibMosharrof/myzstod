import torch
from interpret.masking.base_mask import BaseMask


class MaxMask(BaseMask):

    def get_mask(self, activations, percent):
        mean_vals = torch.mean(torch.abs(activations), dim=0)
        sorted_indices = torch.argsort(mean_vals, descending=True)
        mask_count = int(percent * mean_vals.numel())
        mask = torch.ones_like(mean_vals)
        mask[sorted_indices[:mask_count]] = 0

        return mask
