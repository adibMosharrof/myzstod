from base_datamodule import TodTrainRowCollator
from datamodules.data_collators.base_collator import BaseCollator
import torch


class DecoderCollator(BaseCollator):

    def prepare_item(
        self,
        context_tokens,
        target_tokens,
        pad_tokens,
        target_max_len,
        is_test: bool,
    ) -> TodTrainRowCollator:
        target_unused_len = target_max_len - len(target_tokens)
        target_pad = torch.full([target_unused_len], self.tokenizer.pad_token_id)
        input_items = (
            [pad_tokens, context_tokens]
            if is_test
            else [pad_tokens, context_tokens, target_tokens, target_pad]
        )
        input_tokens = torch.cat(input_items)
        attention_mask = input_tokens.ne(self.tokenizer.pad_token_id).to(torch.int32)
        if is_test:
            label = torch.cat(
                [
                    target_tokens,
                    torch.full([target_unused_len], self.tokenizer.pad_token_id),
                ]
            )
        else:
            label = torch.cat(
                [
                    torch.full(
                        [len(pad_tokens) + len(context_tokens)],
                        self._huggingface_ignore_label_id,
                    ),
                    target_tokens,
                    torch.full([target_unused_len], self._huggingface_ignore_label_id),
                ]
            )
        return TodTrainRowCollator(input_tokens, label, attention_mask)
