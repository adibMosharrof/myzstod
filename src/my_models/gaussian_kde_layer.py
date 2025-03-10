import numpy as np
import torch
from torch import nn
import math


class GaussianKDELayer(nn.Module):
    def __init__(self, samples, bandwidth=None):
        super().__init__()

        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
        elif isinstance(samples, torch.Tensor):
            samples = samples.detach()

        # Register samples as buffer
        self.register_buffer("samples", samples.float().view(-1, 1))

        # Compute or set bandwidth
        if bandwidth is None:
            # Scott's rule
            n = len(samples)
            bandwidth = n ** (-1 / 5) * torch.std(samples)

        self.register_buffer("bandwidth", torch.tensor([bandwidth]).float())

    def forward(self, x):
        x = x.view(-1, 1)
        dist = (x - self.samples.T) ** 2
        kernel = torch.exp(-dist / (2 * self.bandwidth**2))
        kde = kernel.mean(dim=1)
        return kde / (self.bandwidth * math.sqrt(2 * math.pi))
