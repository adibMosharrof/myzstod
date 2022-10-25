from itertools import zip_longest
from typing import Tuple
from matplotlib.cbook import is_math_text
from sentence_transformers import SentenceTransformer, losses, evaluation, InputExample
from transformers import AutoTokenizer, Trainer

from my_enums import ContrastiveConstrants, SpecialTokens
import dstc_utils
import torch


class ContrastiveTrainerHelper:
    tod_tokenizer: AutoTokenizer
    contrastive_model: SentenceTransformer
    token_map: dict[str, int]
    loss_model: None

    def __init__(self, model_path, tokenizer):
        self.contrastive_model = SentenceTransformer(model_path)
        self.tod_tokenizer = tokenizer
        special_tokens = [
            # SpecialTokens.begin_response,
            # SpecialTokens.end_response,
            # SpecialTokens.begin_user_action,
            # SpecialTokens.end_user_action,
            SpecialTokens.begin_action,
            SpecialTokens.end_action,
        ]

        self.token_map = {}
        for token in special_tokens:
            self.token_map[token] = dstc_utils.get_token_id(tokenizer, token)
        self.loss_model = losses.ContrastiveLoss(self.contrastive_model)


class ContrastiveTrainer(Trainer):
    contrastive_helper: ContrastiveTrainerHelper = None

    def compute_loss(self, model, inputs, return_outputs=False):
        out = model(
            **dict((k, inputs[k]) for k in ["input_ids", "attention_mask", "labels"])
        )
        c_model = self.contrastive_helper.contrastive_model
        # tok = c_model.tokenizer
        tok = self.contrastive_helper.tod_tokenizer
        preds = torch.argmax(out.logits, dim=-1)

        sys_act_text = self._get_text_from_tokens(
            preds,
            self.contrastive_helper.token_map[SpecialTokens.begin_action],
            self.contrastive_helper.token_map[SpecialTokens.end_action],
            # SpecialTokens.begin_action,
            # SpecialTokens.end_action,
            # tok,
        )
        user_act_text = tok.batch_decode(
            inputs["contrastive_tokens"], skip_special_tokens=True
        )
        c_inp = []
        for sys, user in zip(sys_act_text, user_act_text):
            c_inp.append(InputExample(texts=[sys, user], label=1.0))
        features, labels = c_model.smart_batching_collate(c_inp)

        contrastive_loss = self.contrastive_helper.loss_model(features, labels)
        combined_loss = contrastive_loss + out.loss
        return (combined_loss, out.logits) if return_outputs else combined_loss
        # return (out.loss, out.logits) if return_outputs else out.loss

    def _get_text_from_tokens(
        self,
        data: torch.Tensor,
        s_id: int,
        e_id: int,
    ) -> list[str]:
        input_ids = []
        for row in data:
            try:
                s_index = (row == s_id).nonzero(as_tuple=True)[0][0]
                e_index = (row == e_id).nonzero(as_tuple=True)[0][0]
            except IndexError:
                input_ids.append(torch.tensor([], device="cuda"))
                continue
            if s_index is None or e_index is None or s_index > e_index:
                input_ids.append(torch.tensor([], device="cuda"))
                continue
            ind = torch.arange(s_index + 1, e_index, device="cuda")
            input_ids.append(row.index_select(0, ind))
        return input_ids

    def _get_text_from_tokens_asd(
        self,
        data: torch.Tensor,
        start_token: SpecialTokens,
        end_token: SpecialTokens,
        tok: AutoTokenizer,
    ) -> list[str]:
        # input_ids = []
        # s_id = self.contrastive_helper.token_map[start_token]
        # e_id = self.contrastive_helper.token_map[end_token]
        # # try:
        # s_indexes = (data == s_id).nonzero(as_tuple=True)[1]
        # e_indexes = (data == e_id).nonzero(as_tuple=True)[1]

        # # for item in data:
        # # try:
        # #     s_index = (data == s_id).nonzero(as_tuple=False)[0]
        # #     e_index = (data == e_id).nonzero(as_tuple=False)[0]
        # # except IndexError:
        # # input_ids.append(torch.tensor([], device="cuda"))
        # # return ""
        # for s_index, e_index, row in zip_longest(
        #     s_indexes, e_indexes, data
        # ):  # need to work in this next
        #     input_ids.append(row[s_index : e_index + 1])
        #     if s_index is None or e_index is None or s_index > e_index:
        #         input_ids.append(torch.tensor([], device="cuda"))
        #         continue
        #     ind = torch.arange(s_index, e_index, device="cuda")
        #     input_ids.append(data.index_select(0, ind))
        # return tok.batch_decode(input_ids)
        # return {
        #     "input_ids": torch.nn.utils.rnn.pad_sequence(
        #         input_ids, batch_first=True, padding_value=tok.pad_token_id
        #     ),
        #     "attention_mask": torch.nn.utils.rnn.pad_sequence(
        #         att_masks, batch_first=True, padding_value=0
        #     ),
        # }
        input_ids = []
        s_id = self.contrastive_helper.token_map[start_token]
        e_id = self.contrastive_helper.token_map[end_token]
        s_indexes = (data == s_id).nonzero(as_tuple=True)[1]
        e_indexes = (data == e_id).nonzero(as_tuple=True)[1]

        mask = []
        for s_index, e_index, row in zip_longest(
            s_indexes, e_indexes, data, fillvalue=None
        ):
            if s_index is None:
                s_index = -1
            if e_index is None:
                e_index = 0

        for s_index, e_index, row in zip_longest(s_indexes, e_indexes, data):
            input_ids.append(row[s_index : e_index + 1])
            if s_index is None or e_index is None or s_index > e_index:
                input_ids.append(torch.tensor([], device="cuda"))
                continue
            ind = torch.arange(s_index, e_index, device="cuda")
            input_ids.append(data.index_select(0, ind))
        return tok.batch_decode(input_ids)


class ContrastiveUtils:
    @classmethod
    def _get_tokens_from_contrast_with(
        self, contrast_with: str
    ) -> Tuple[SpecialTokens, SpecialTokens, bool]:
        multiple_values = False
        if contrast_with == ContrastiveConstrants.USER_ACT:
            start_token = SpecialTokens.begin_user_action
            end_token = SpecialTokens.end_user_action
            multiple_values = True
        elif contrast_with == ContrastiveConstrants.NLG:
            start_token = SpecialTokens.begin_response
            end_token = SpecialTokens.end_response
        return start_token, end_token, multiple_values
