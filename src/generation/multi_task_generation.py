from typing import Union

from dotmap import DotMap
import torch
from generation.generation_base import GenerationBase
from my_enums import MultiTaskNames, SpecialTokens

from simple_tod_dataclasses import TodTestDataBatch
from transformers import AutoModel, AutoTokenizer
from torch import Tensor


class MultiTaskGeneration(GenerationBase):
    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        task_names: list[MultiTaskNames],
    ):
        super().__init__(model, tokenizer)
        self.task_names = task_names

    def _get_generation(self, batch, max_len: int) -> list[Tensor]:
        gens = []
        for task in self.task_names:
            self.model.set_adapter(task.value)
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                gen = self.model.generate(
                    inputs=batch.input_ids,
                    attention_mask=batch.attention_masks,
                    max_length=max_len,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            gens.append(gen)
        return gens

    def remove_context(
        self, gen: list[Tensor], context_len: int, max_len: int
    ) -> list[Tensor]:
        out = torch.full(
            [len(gen), gen[0].shape[0], max_len - context_len],
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.int,
            device=gen[0].device,
        )
        for i, g in enumerate(gen):
            pred_len = g.shape[1] - context_len
            end_idx = context_len + pred_len
            out[i, :, :pred_len] = g[:, context_len:end_idx]

        # print("Gen out")
        # print("dst")
        # print(self.tokenizer.decode(gen[0][0]).replace(SpecialTokens.pad_token, ""))
        # print("action")
        # print(self.tokenizer.decode(gen[1][0]).replace(SpecialTokens.pad_token, ""))
        # print("with context removed")
        # print("dst")
        # print(self.tokenizer.decode(out[0][0]).replace(SpecialTokens.pad_token, ""))
        # print("action")
        # print(self.tokenizer.decode(out[1][0]).replace(SpecialTokens.pad_token, ""))
        return out

    def hook_before_remove_pad(self, gen: list[Tensor]) -> list[Tensor]:
        # do it manually first in a for loop, then try to do it with torch function
        out = torch.empty(
            [gen.shape[1], gen.shape[0] * gen.shape[2]],
            dtype=torch.int,
            device=gen.device,
        )
        for i, g in enumerate(gen):
            out[:, i * gen.shape[2] : (i + 1) * gen.shape[2]] = g
        return out
        # return gen.reshape(-1)
