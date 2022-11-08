from typing import Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, losses
from transformers import AutoTokenizer, Trainer
from torch.nn.utils.rnn import pad_sequence
from my_enums import ContrastiveConstrants, SpecialTokens
import dstc_utils
import torch


class ContrastiveTrainerHelper:
    tod_tokenizer: AutoTokenizer
    contrastive_model: SentenceTransformer
    token_map: dict[str, int]
    loss_model: None
    max_token_len: int = None

    def __init__(self, model_path, tokenizer, max_token_len):
        self.contrastive_model = SentenceTransformer(model_path).cuda()
        self.tod_tokenizer = tokenizer
        special_tokens = [
            SpecialTokens.begin_response,
            SpecialTokens.end_response,
            SpecialTokens.begin_user_action,
            SpecialTokens.end_user_action,
            SpecialTokens.begin_action,
            SpecialTokens.end_action,
        ]
        self.max_token_len = max_token_len

        self.token_map = {}
        for token in special_tokens:
            self.token_map[token] = dstc_utils.get_token_id(tokenizer, token)
        self.loss_model = losses.CosineSimilarityLoss(self.contrastive_model)


class ContrastiveTrainer(Trainer):
    contrastive_helper: ContrastiveTrainerHelper = None

    def compute_loss(self, model, inputs, return_outputs=False):
        out = model(
            **dict((k, inputs[k]) for k in ["input_ids", "attention_mask", "labels"])
        )
        # tok = c_model.tokenizer
        tok = self.contrastive_helper.tod_tokenizer
        preds = torch.argmax(out.logits, dim=-1)

        # sys_act_tokens = self._get_text_from_tokens_qwe(
        # sys_feats = self._get_sys_feats(
        sys_feats = self._get_text_from_tokens_asd(
            preds,
            self.contrastive_helper.token_map[SpecialTokens.begin_action],
            self.contrastive_helper.token_map[SpecialTokens.end_action],
            tok.pad_token_id,
        )

        # user_feats = self._get_sys_feats(
        user_feats = self._get_text_from_tokens_asd(
            preds,
            self.contrastive_helper.token_map[SpecialTokens.begin_user_action],
            self.contrastive_helper.token_map[SpecialTokens.end_user_action],
            tok.pad_token_id,
        )

        labels = torch.ones([inputs["input_ids"].shape[0]], device="cuda")
        contrastive_loss = self.contrastive_helper.loss_model(
            [sys_feats, user_feats], labels
        )
        combined_loss = contrastive_loss + out.loss
        return (combined_loss, out.logits) if return_outputs else combined_loss
        # return (out.loss, out.logits) if return_outputs else out.loss

    def get_features(self, sys_acts, user_acts):
        features = [[s, u] for s, u in zip(sys_acts, user_acts)]
        labels = [1.0 for _ in range(len(features))]
        return features, labels

    def _get_sys_feats(
        self, data: torch.Tensor, s_id: int, e_id: int, pad_id: int
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
        padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id).int()
        fill = torch.full(
            [padded.shape[0], self.contrastive_helper.max_token_len - padded.shape[1]],
            pad_id,
            device="cuda",
            dtype=int,
        )
        out = torch.cat([padded, fill], dim=1)
        mask = torch.cat(
            [
                torch.full_like(padded, 1, dtype=int),
                torch.full_like(fill, 0, dtype=int),
            ],
            dim=1,
        )
        return {"input_ids": out, "attention_mask": mask}
        # return input_ids

    def _get_text_from_tokens_asd(
        self,
        data: torch.Tensor,
        s_id: int,
        e_id: int,
        pad_id: int,
    ) -> torch.Tensor:
        # start_count = data == s_id
        # end_count = data == e_id
        # invalid = (start_count.sum(axis=1) - end_count.sum(axis=1)) != 0
        # data[invalid] = pad_id
        row_length = data.shape[1]
        start_idx = (data == s_id).long().argmax(axis=1)
        start_mask = (start_idx[:, None] - torch.arange(row_length, device="cuda")) >= 0
        data[start_mask] = pad_id
        end_idx = row_length - (data == e_id).long().argmax(axis=1)
        end_mask = (
            end_idx[:, None] + torch.arange(row_length, device="cuda")
        ) >= row_length
        data[end_mask] = pad_id
        return self._pad_mask_data(data, pad_id)

    def _pad_mask_data(self, data: torch.Tensor, pad_id: int) -> torch.Tensor:
        input_ids = torch.full(
            [data.shape[0], self.contrastive_helper.max_token_len],
            pad_id,
            device="cuda",
        )
        att_mask = torch.zeros_like(input_ids, dtype=int, device="cuda")
        for i, r in enumerate(data):
            row = r[r != pad_id]
            input_ids[i, : len(row)] = row
            att_mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": att_mask}

    def _get_text_from_tokens_qwe(
        self, data: torch.Tensor, s_id: int, e_id: int, pad_id: int
    ) -> list[str]:
        zeros = torch.full(data.shape, 0, device="cuda")
        row_length = data.shape[1]
        # start_idx = (data == s_id).long().argmax(axis=1)
        start_mask = torch.where(data == s_id, data, zeros)
        data[start_mask] = pad_id
        end_idx = row_length - (data == e_id).long().argmax(axis=1)
        end_mask = (
            end_idx[:, None] + torch.arange(row_length, device="cuda")
        ) >= row_length
        data[end_mask] = pad_id

        input_ids = torch.empty_like(data)
        att_mask = torch.empty_like(data)
        for i, r in enumerate(data):
            row = r[r != pad_id]
            input_ids[i] = torch.cat([row, torch.full([row_length - len(row)], pad_id)])
            att_mask[i] = torch.cat(
                [torch.full([len(row)], 1), torch.full([row_length - len(row)], 0)]
            )
        return {"input_ids": input_ids, "attention_mask": att_mask}


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


@dataclass
class ContrastiveTokens:
    start_token: SpecialTokens
    end_token: SpecialTokens
