from torch import nn
import torch
from scipy import stats

from my_models.gaussian_kde_layer import GaussianKDELayer


class MaskLayer(nn.Module):
    def __init__(self, lower_bound, upper_bound, replacement_values):
        super(MaskLayer, self).__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.replacement_values = replacement_values

        # Initialize KDE related attributes
        self.kde_models_by_class = {}  # Dictionary to store KDEs for each class
        self.kde_layers_by_class = nn.ModuleDict()  # Store GPU KDE layers
        self.prob_threshold = None
        self.use_probability = False
        self.current_class = None
        self.kde_layers = None

    def fit_kde(self, activation_lists, threshold=0.3):
        """
        Set up KDE models for probability-based masking for each class

        Args:
            activation_lists: List of tensors, shape [num_classes][num_examples, num_neurons]
            threshold: Probability threshold for masking
        """
        num_classes = len(activation_lists)
        feature_dim = activation_lists[0].shape[1]  # Should be 768

        # Fit KDE for each class separately
        for class_idx in range(num_classes):
            class_kdes = []  # CPU KDEs (keeping for backwards compatibility)
            class_kde_layers = nn.ModuleList()  # GPU KDE layers
            class_data = activation_lists[class_idx]

            # For each neuron
            for feat_idx in range(feature_dim):
                # Get all examples for this neuron in this class
                neuron_values = class_data[:, feat_idx]
                if isinstance(neuron_values, torch.Tensor):
                    neuron_values = neuron_values.detach().cpu().numpy()

                # Fit CPU KDE for backwards compatibility
                kde = stats.gaussian_kde(neuron_values)
                class_kdes.append(kde)

                # Create GPU KDE layer
                kde_layer = GaussianKDELayer(neuron_values)
                class_kde_layers.append(kde_layer)

            self.kde_models_by_class[class_idx] = class_kdes
            self.kde_layers_by_class[str(class_idx)] = class_kde_layers

        self.prob_threshold = threshold
        self.use_probability = True

    def set_class(self, class_idx):
        """Set which class's KDE to use for masking"""
        if class_idx not in self.kde_models_by_class:
            raise ValueError(f"No KDE models fitted for class {class_idx}")
        self.current_class = class_idx
        self.kde_models = self.kde_models_by_class[class_idx]
        self.kde_layers = self.kde_layers_by_class[str(class_idx)]

    def forward(self, x):
        if not self.use_probability:
            # Original bounds-based masking
            lower_bound = self.lower_bound.to(dtype=x.dtype, device=x.device).view(
                1, 1, -1
            )
            upper_bound = self.upper_bound.to(dtype=x.dtype, device=x.device).view(
                1, 1, -1
            )
            replacement_values = self.replacement_values.to(
                dtype=x.dtype, device=x.device
            ).view(1, 1, -1)
            mask = (x >= lower_bound) & (x <= upper_bound)
            x = torch.where(mask, replacement_values, x)
            return x
        else:
            if self.current_class is None:
                raise ValueError(
                    "Must call set_class before using probability-based masking"
                )

            batch_size, seq_len, feature_dim = x.shape

            # Process on GPU using KDE layers
            x_flat = x.reshape(-1, feature_dim)

            # Calculate probabilities for all features
            all_probs = torch.zeros(
                x_flat.shape[0], feature_dim, device=x.device, dtype=torch.float32
            )

            # Process features using GPU KDE layers
            for i, kde_layer in enumerate(self.kde_layers):
                all_probs[:, i] = kde_layer(x_flat[:, i])

            # Reshape probabilities back to original shape
            all_probs = all_probs.reshape(batch_size, seq_len, feature_dim)

            # Create mask
            mask = all_probs >= self.prob_threshold

            # Apply mask using replacement values
            replacement_values = self.replacement_values.to(
                dtype=x.dtype, device=x.device
            ).view(1, 1, -1)
            x = torch.where(mask, replacement_values, x)

            return x

    def set_perms(self, lower_bound, upper_bound, replacement_values):
        """Set parameters and switch to bounds-based masking"""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.replacement_values = replacement_values
        self.use_probability = False

    def set_probability_threshold(self, threshold):
        """Update probability threshold"""
        self.prob_threshold = threshold
