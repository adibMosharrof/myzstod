from interpret.masking.base_mask import BaseMask
from interpret.masking.mask_model_base import MaskModelBase


class InterventionBase:
    def __init__(
        self, cfg, mask: BaseMask, mask_percent: float, mask_model_type: MaskModelBase
    ):
        self.cfg = cfg
        self.mask = mask
        self.mask_percent = mask_percent
        self.mask_model_type = mask_model_type

    def intervene(self, model, activations):
        raise NotImplementedError

    def _get_intervention_name(self):
        raise NotImplementedError

    def get_intervention_name(self, feature_name, feature_domain):
        name = self._get_intervention_name()
        return f"{name}_{feature_name}_{feature_domain}"
