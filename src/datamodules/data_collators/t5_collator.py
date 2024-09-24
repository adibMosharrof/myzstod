from base_datamodule import TodTrainRowCollator
from datamodules.data_collators.base_collator import BaseCollator
import torch


class T5Collator(BaseCollator):

    def prepare_item(
        self,
        context_tokens,
        target_tokens,
        pad_tokens,
        target_max_len,
    ) -> TodTrainRowCollator:
        input_tokens = torch.cat(
            [
                pad_tokens,
                context_tokens,
            ]
        )
        attention_mask = input_tokens.ne(self.tokenizer.pad_token_id).to(torch.int)

        target_unused_len = target_max_len - len(target_tokens)
        if target_unused_len < 0:
            raise Exception("Target is too long")
        label = torch.cat(
            [
                target_tokens,
                torch.full([target_unused_len], self.tokenizer.pad_token_id),
                torch.full([1], self.tokenizer.eos_token_id),
            ]
        )
        return TodTrainRowCollator(input_tokens, label, attention_mask)
