from typing import Union

from dotmap import DotMap
import torch
from dstc import dstc_utils
from generation.generation_base import GenerationBase

from my_enums import SpecialTokens
from simple_tod_dataclasses import TodTestDataBatch


class SimpleGeneration(GenerationBase):
    def _get_generation(self, batch, min_len: int, max_len: int):

        # gen = self.model.generate(
        #     inputs=batch.input_ids,
        #     attention_mask=batch.attention_masks,
        #     max_length=max_len,
        #     # min_length=max_len,
        #     eos_token_id=self.tokenizer.eos_token_id,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     use_cache=True,
        #     # bos_token_id=self.tokenizer.bos_token_id,
        # )
        gen = self.model.generate(
            inputs=batch.input_ids,
            attention_mask=batch.attention_masks,
            max_length=max_len,
            do_sample=False,
            use_cache=True,
            top_k=50,
            top_p=0.92,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.pad_gen_to_max_len(gen, max_len)

    def remove_context(self, gen, context_len: int, max_len: int):
        return gen[:, context_len:]

    def postprocess_generation(self, batch: list[str]) -> list[str]:
        target_responses = [
            dstc_utils.get_text_in_between(
                row, SpecialTokens.begin_response, SpecialTokens.end_response, ""
            )
            for row in batch
        ]
        return target_responses
