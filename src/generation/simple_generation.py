from typing import Union

from dotmap import DotMap
import torch
from generation.generation_base import GenerationBase

from simple_tod_dataclasses import TodTestDataBatch


class SimpleGeneration(GenerationBase):
    def _get_generation(self, batch, max_len: int):
        # with torch.cuda.amp.autocast():
        #     gen = self.model.generate(
        #         inputs=batch.input_ids,
        #         attention_mask=batch.attention_masks,
        #         max_length=max_len,
        #         eos_token_id=self.tokenizer.eos_token_id,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         bos_token_id=self.tokenizer.bos_token_id,
        #     )
        # with torch.no_grad():
        with torch.cuda.amp.autocast():
            gen = self.model.generate(
                inputs=batch.input_ids,
                attention_mask=batch.attention_masks,
                max_length=max_len,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                # bos_token_id=self.tokenizer.bos_token_id,
            )
        return gen

    def remove_context(self, gen, context_len: int, max_len: int):
        return gen[:, context_len:]
