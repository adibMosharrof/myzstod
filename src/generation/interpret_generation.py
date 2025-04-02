from tqdm import tqdm
from generation.simple_generation import SimpleGeneration
from utilities import nethook, tensor_utilities
import torch


class InterpretGeneration(SimpleGeneration):

    def get_generation(
        self,
        batch,
        min_len: int,
        max_len: int,
        context_len,
        should_post_process,
        accelerator,
        metric_manager,
    ):
        batch_gpu = self.move_to_gpu(batch, accelerator)
        generated_tokens, confidences, fc_vals = self._get_generation(
            batch_gpu, min_len, max_len, accelerator
        )
        all_seq_len = torch.tensor(
            [generated_tokens.shape[1]], device=accelerator.device
        )
        max_seq_len = accelerator.gather(all_seq_len).max().item()
        # generated_tokens = self.pad_to_len(generated_tokens, max_seq_len)
        # confidences = self.pad_to_len(confidences, max_seq_len)
        # fc_vals = self.pad_to_len(fc_vals, max_seq_len)
        generated_tokens = tensor_utilities.pad_to_len(
            generated_tokens, max_seq_len, self.tokenizer.pad_token_id
        )
        confidences = tensor_utilities.pad_to_len(
            confidences, max_seq_len, self.tokenizer.pad_token_id
        )
        fc_vals = tensor_utilities.pad_to_len(
            fc_vals, max_seq_len, self.tokenizer.pad_token_id
        )
        # accelerator.wait_for_everyone()
        g_t, c, f = accelerator.gather_for_metrics(
            (generated_tokens, confidences, fc_vals)
        )
        return g_t, c, f
        return generated_tokens, confidences, fc_vals

    # def pad_gen_to_max_len(self, gen, max_len):
    #     return super().pad_gen_to_max_len(gen, max_len)

    def _get_generation(
        self, batch, min_len: int, max_len: int, accelerator, no_repeat_ngram_size=3
    ):
        batch_size = batch.input_ids.size(0)
        all_fc_vals = []
        all_confidences = []
        all_generated_tokens = []
        prompt_ids = batch.input_ids
        prompt_attention_masks = batch.attention_masks
        active_sequences = torch.ones(
            batch_size, dtype=torch.bool, device=accelerator.device
        )

        with torch.no_grad():
            for _ in tqdm(range(min_len), desc="token generation"):
                # for _ in range(min_len):
                if (
                    not active_sequences.any()
                ):  # Stop early if all sequences reached EOS
                    break
                with nethook.TraceDict(self.model, ["transformer.mask_layer"]) as ret:
                    outputs = self.model(
                        input_ids=prompt_ids,
                        attention_mask=prompt_attention_masks,
                    )
                    fc1_vals = [
                        ret[layer_fc1_vals].output[:, -1, :] for layer_fc1_vals in ret
                    ]
                    # Shape: [batch, hidden_dim]
                    all_fc_vals.append(torch.cat(fc1_vals, dim=0))
                next_token_logits = outputs.logits[:, -1, :]

                next_tokens = torch.argmax(next_token_logits, dim=-1)
                confidences = torch.nn.functional.softmax(next_token_logits, dim=-1)
                all_generated_tokens.append(next_tokens)
                all_confidences.append(confidences)
                prompt_ids = torch.cat([prompt_ids, next_tokens.unsqueeze(-1)], dim=1)
                prompt_attention_masks = torch.cat(
                    [
                        prompt_attention_masks,
                        torch.ones_like(next_tokens).unsqueeze(-1),
                    ],
                    dim=1,
                )
                active_sequences &= (
                    next_tokens.squeeze(-1) != self.tokenizer.eos_token_id
                )
                next_tokens[~active_sequences] = (
                    self.tokenizer.eos_token_id  # Force EOS for completed sequences
                )
        all_fc_vals = torch.stack(
            all_fc_vals, dim=1
        )  # Shape: [batch, seq_len, hidden_dim]
        all_confidences = torch.stack(all_confidences, dim=1)  # Shape: [batch, seq_len]
        all_generated_tokens = torch.stack(
            all_generated_tokens, dim=1
        )  # Shape: [batch, seq_len]
        return all_generated_tokens, all_confidences, all_fc_vals

    def pad_to_len(self, tensor, max_len, pad_value=None):
        pad_value = pad_value or self.tokenizer.pad_token_id
        pad_size = max_len - tensor.shape[1]
        if pad_size <= 0:
            return tensor  # No padding needed

        pad_shape = list(tensor.shape)
        pad_shape[1] = pad_size

        pad_tensor = torch.full_like(
            tensor[:, :pad_size], pad_value
        )  # Uses same dtype & device
        return torch.cat([tensor, pad_tensor], dim=1)
