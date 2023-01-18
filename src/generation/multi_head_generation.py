from typing import Union

from dotmap import DotMap
import torch
from generation.generation_base import GenerationBase

from simple_tod_dataclasses import TodTestDataBatch


class MultiHeadGeneration(GenerationBase):

    def move_to_gpu(self, batch:  TodTestDataBatch):
        batch_gpu = DotMap()
        field_names = ["attention_masks", "input_ids"]
        for name in field_names:
            for head_name, value in getattr(batch, name).items():
                batch_gpu[name][head_name] = value.cuda()
        return batch_gpu
    
    def _get_generation(self, batch, max_len:int):
        gen = self.model.generate(
            inputs=batch.input_ids,
            attention_mask=batch.attention_masks,
            max_length=max_len,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        return gen
        # return [g for g in gen]
       
    def remove_context(self, gen, context_len:int):
        return [row[:,context_len:] for row in gen]

    def hook_before_remove_pad(self, gen):
        return torch.cat(gen, dim=1)

    # def remove_padding(self, gen):
    #     no_pad = [super(__class__, self).remove_padding(row) for row in gen]
    #     return no_pad
        # return np.concatenate(no_pad, axis=0)
            