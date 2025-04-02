import numpy as np
from interpret.interpret_utilities import InterpretUtilities
from interpret.interventions.intervention_base import InterventionBase


class SimpleIntervention(InterventionBase):

    def intervene(self, model, activations, interpret_datasets, index):
        masked_activations = self.mask.get_mask(
            activations[index], self.cfg.percent_mask
        )
        self.mask_model_type.mask(model, masked_activations)
        return model

    def _get_intervention_name(self):
        return "simple"
