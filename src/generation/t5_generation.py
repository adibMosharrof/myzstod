from typing import Union

from dotmap import DotMap
import torch
from generation.generation_base import GenerationBase

from simple_tod_dataclasses import TodTestDataBatch


class T5Generation(GenerationBase):
    def _get_generation(self, batch, min_len: int, max_len: int):
        gen = self.model.generate(
            inputs=batch.input_ids,
            attention_mask=batch.attention_masks,
            max_length=min_len,
            # min_length=max_len,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            # bos_token_id=self.tokenizer.bos_token_id,
        )
        # return gen
        return self.pad_gen_to_max_len(gen, min_len)

    def remove_context(self, gen, context_len: int, max_len: int):
        return gen
