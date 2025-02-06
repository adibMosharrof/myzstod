from base_datamodule import TodTrainRowCollator
from datamodules.data_collators.decoder_collator import DecoderCollator
from tod.turns.turn_csv_row_base import TurnCsvRowBase
from utilities.tokenizer_utilities import TokenizerUtilities
import torch


class CrossCollator(DecoderCollator):

    def __init__(
        self,
        tokenizer,
        nlg_prompt_cls,
        max_token_len=1024,
        test_prompt_max_len=750,
        schema_max_length=350,
    ):
        super().__init__(tokenizer, nlg_prompt_cls, max_token_len, test_prompt_max_len)
        self.schema_max_length = schema_max_length

    def collate_single_item(
        self, item: TurnCsvRowBase, target_max_len: int, is_test: bool = False
    ) -> TodTrainRowCollator:
        schema_prompt = self.nlg_prompt_cls.get_schema_prompt(item.domains, item.schema)
        generation_prompt = self.nlg_prompt_cls.get_generation_prompt(
            item.domains, item.context
        )
        schema_ids, schema_attention_mask = TokenizerUtilities.tokenize_with_pad(
            text=schema_prompt, tokenizer=self.tokenizer, max_len=self.schema_max_length
        )
        context_tokens = TokenizerUtilities.tokenize(
            text=generation_prompt,
            tokenizer=self.tokenizer,
            max_len=self.test_prompt_max_len,
        )
        context_unused_len = self.test_prompt_max_len - len(context_tokens)
        if context_unused_len < 0:
            context_tokens = self.trim_dialog_history(item, -context_unused_len)
            context_unused_len = self.test_prompt_max_len - len(context_tokens)
        pad = torch.full([context_unused_len], self.tokenizer.pad_token_id)
        target_tokens = self.get_target_tokens(item, target_max_len)
        decoder_item = self.prepare_item(
            context_tokens=context_tokens,
            target_tokens=target_tokens,
            pad_tokens=pad,
            target_max_len=target_max_len,
            is_test=is_test,
        )
        return TodTrainRowCollator(
            schema_ids=schema_ids,
            schema_attention_mask=schema_attention_mask,
            **decoder_item
        )
