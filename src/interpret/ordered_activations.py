import numpy as np
import torch


class OrderedActivations:
    def __init__(self, cfg):
        self.cfg = cfg

    def find_subtensor_indices(self, batch_tensor, pattern):
        batch_size, seq_len = batch_tensor.shape  # Batch size and sequence length
        pattern = torch.tensor(
            pattern, dtype=batch_tensor.dtype, device=batch_tensor.device
        )
        pat_len = pattern.size(0)

        # Create a sliding window over the last dimension (seq_len)
        windows = batch_tensor.unfold(
            1, pat_len, 1
        )  # Shape: (batch_size, seq_len - pat_len + 1, pat_len)

        # Compare each window with the pattern
        matches = (windows == pattern.unsqueeze(0)).all(
            dim=2
        )  # Shape: (batch_size, seq_len - pat_len + 1)

        # Get indices where the match occurs for each sequence in the batch
        match_indices = [match.nonzero(as_tuple=True)[0] + pat_len for match in matches]

        return match_indices  # List of lists, one per batch item

    def get_first_and_other_slot_indices(self, generated_tokens, tokenizer):
        first_slot_prefix = "='"
        other_slot_prefix = "', '"
        first_slot_prefix_tokens = tokenizer.encode(first_slot_prefix)
        other_slot_prefix_tokens = tokenizer.encode(other_slot_prefix)

        other_slot_prefix_indices = self.find_subtensor_indices(
            generated_tokens, other_slot_prefix_tokens
        )
        first_slot_prefix_indices = self.find_subtensor_indices(
            generated_tokens, first_slot_prefix_tokens
        )
        return first_slot_prefix_indices, other_slot_prefix_indices

    def get_activations_and_confidences(
        self,
        interpret_text,
        generated_tokens,
        fc_vals,
        confidences,
        tokenizer,
        accelerator,
    ):
        first_slot_prefix_indices, other_slot_prefix_indices = (
            self.get_first_and_other_slot_indices(generated_tokens, tokenizer)
        )
        # first_slot_activations = fc_vals[
        #     torch.arange(len(first_slot_prefix_indices)),
        #     first_slot_prefix_indices,
        # ]
        first_slot_activations = []
        for batch_index, index in enumerate(first_slot_prefix_indices):
            if len(index):
                first_slot_activations.append(fc_vals[batch_index, index[0]])

        max_slots = max([len(row) for row in other_slot_prefix_indices])
        other_slot_activations = [[] for _ in range(max_slots)]
        for batch_index, batch_slot_indices in enumerate(other_slot_prefix_indices):
            for slot_num, index in enumerate(batch_slot_indices):
                other_slot_activations[slot_num].append(fc_vals[batch_index, index])
        all_activations = [first_slot_activations] + other_slot_activations
        return all_activations, confidences
        a = 1

    def concat_all_batch_activations(self, all_activations):
        max_num_slots = max([len(row) for row in all_activations])
        out = [[] for _ in range(max_num_slots)]
        for row in all_activations:
            for slot_num, activations in enumerate(row):
                act = [a.cpu().numpy() for a in activations]
                out[slot_num].extend(act)
        out_np = [np.array(row) for row in out]
        return out_np
