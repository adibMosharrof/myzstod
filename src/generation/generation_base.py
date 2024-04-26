from abc import ABC, abstractmethod
from typing import Union

from dotmap import DotMap
import torch
from my_enums import SpecialTokens
from transformers import AutoModel, AutoTokenizer
from simple_tod_dataclasses import NlgTestDataBatch, TodTestDataBatch
from torch import Tensor


class GenerationBase(ABC):
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def _get_generation(
        self, batch: TodTestDataBatch, max_len: int
    ) -> Union[Tensor, list[Tensor]]:
        pass

    def move_to_gpu(self, batch: TodTestDataBatch, accelerator):
        batch_gpu = DotMap()
        batch_gpu.input_ids = batch.input_ids.to(accelerator.device)
        batch_gpu.attention_masks = batch.attention_masks.to(accelerator.device)
        # batch_gpu.contexts_text = batch.contexts_text.to(accelerator.device)
        # batch_gpu.targets_text = batch.targets_text.to(accelerator.device)
        batch_gpu.dialog_ids = batch.dialog_ids.to(accelerator.device)
        batch_gpu.turn_ids = batch.turn_ids.to(accelerator.device)
        batch_gpu.labels = batch.labels.to(accelerator.device)
        batch_gpu.turn_row_types = batch.turn_row_types.to(accelerator.device)
        batch_gpu.is_retrievals = batch.is_retrievals.to(accelerator.device)
        batch_gpu.is_slot_fills = batch.is_slot_fills.to(accelerator.device)
        batch_gpu.domain_ids = batch.domain_ids.to(accelerator.device)

        return batch_gpu

    @abstractmethod
    def remove_context(self, gen: Union[Tensor, list[Tensor]], context_len: int):
        pass

    def remove_pad_decode(self, text, skip_special_tokens=False):
        if skip_special_tokens:
            return self.tokenizer.batch_decode(
                text, skip_special_tokens=skip_special_tokens
            )
        txt_no_pad = self.remove_padding(text)
        return self.tokenizer.batch_decode(txt_no_pad, skip_special_tokens=False)

    def get_generation(
        self,
        batch: Union[TodTestDataBatch, NlgTestDataBatch],
        min_gen_len: int,
        max_len: int,
        context_len: int,
        should_post_process: bool,
        accelerator,
        metric_manager,
    ) -> list[str]:
        batch_gpu = self.move_to_gpu(batch, accelerator)
        # batch_gpu = batch
        curr_gen = self._get_generation(batch_gpu, min_gen_len, max_len)
        gen_without_context = self.remove_context(curr_gen, context_len, max_len)
        # gen_after_hook = self.hook_before_remove_pad(gen_without_context)
        # gen_after_hook = gen_without_context
        (
            gen_after_hook,
            # target,
            # contexts,
            dialog_ids,
            turn_ids,
            domain_ids,
            input_tokens,
            turn_row_types,
            label_tokens,
            is_retrievals,
            is_slot_fills,
        ) = accelerator.gather_for_metrics(
            (
                gen_without_context,
                # batch_gpu.targets_text,
                # batch_gpu.contexts_text,
                batch_gpu.dialog_ids,
                batch_gpu.turn_ids,
                batch_gpu.domain_ids,
                batch_gpu.input_ids,
                batch_gpu.turn_row_types,
                batch_gpu.labels,
                batch_gpu.is_retrievals,
                batch_gpu.is_slot_fills,
            )
        )
        # gen_without_context = self.remove_context(gen, context_len, max_len)
        # gen_after_hook = self.hook_before_remove_pad(gen_without_context)
        # if accelerator.is_main_process:
        gen_txt = self.remove_pad_decode(gen_after_hook)
        # target_txt = self.remove_pad_decode(target)
        # dialog_ids = self.remove_pad_decode(dialog_ids, skip_special_tokens=True)
        # turn_ids = self.remove_pad_decode(turn_ids, skip_special_tokens=True)
        domains = self.remove_pad_decode(domain_ids, skip_special_tokens=True)

        metric_manager.add_batch(
            input_tokens,
            label_tokens,
            gen_after_hook,
            turn_row_types,
            is_retrievals,
            is_slot_fills,
            dialog_ids,
            turn_ids,
            domains,
        )
        if should_post_process:
            gen_txt = self.postprocess_generation(gen_txt)
        # return target_txt, gen_txt, contexts, dialog_ids, turn_ids
        return gen_txt, dialog_ids, turn_ids

    def hook_before_remove_pad(self, gen: Union[Tensor, list[Tensor]]):
        return gen

    def postprocess_generation(self, batch: list[str]) -> list[str]:
        out = []
        required_tokens = [
            SpecialTokens.begin_target,
            SpecialTokens.begin_dsts,
            SpecialTokens.begin_dst,
            SpecialTokens.begin_intent,
        ]
        for item in batch:
            text_to_add = [rt for rt in required_tokens if rt not in item]
            out_text = "".join(["".join(text_to_add), item])
            out.append(out_text)
        return out

    def remove_padding(self, gen):
        return [row[row != self.tokenizer.pad_token_id] for row in gen]

    def pad_gen_to_max_len(self, gen, max_len: int):
        pad_amount = max_len - gen.shape[1]
        pad = torch.full(
            [gen.shape[0], pad_amount],
            fill_value=self.tokenizer.pad_token_id,
            dtype=torch.int,
            device=gen.device,
        )
        out = torch.hstack([gen, pad])
        return out
