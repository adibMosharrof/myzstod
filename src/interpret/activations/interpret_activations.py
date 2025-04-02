import torch

from interpret.activations.base_activations import BaseActivations


class InterpretActivation(BaseActivations):

    def get_mask_and_indices(
        self, interpret_text, generated_tokens, tokenizer, accelerator
    ):
        target_token_ids = torch.tensor(
            tokenizer.encode(interpret_text), device=accelerator.device
        )
        target_len = len(target_token_ids)

        rolling_windows = generated_tokens.unfold(1, target_len, 1)
        matches = (rolling_windows == target_token_ids).all(dim=2)
        indices = torch.where(
            matches,
            torch.arange(matches.shape[1], device=accelerator.device).expand_as(
                matches
            ),
            torch.full_like(matches, -1),
        )
        first_indices = torch.max(indices, dim=1).values
        mask = first_indices != -1
        return mask, first_indices

    def get_activations_and_confidences(
        self,
        interpret_text,
        generated_tokens,
        fc_vals,
        confidences,
        tokenizer,
        accelerator,
    ):
        mask, indices = self.get_mask_and_indices(
            interpret_text, generated_tokens, tokenizer, accelerator
        )
        activations = fc_vals[mask, indices]
        confidences = confidences[mask, indices]
        return activations, confidences

    def concat_all_batch_activations(self, all_activations):
        return torch.cat(all_activations, dim=0)
