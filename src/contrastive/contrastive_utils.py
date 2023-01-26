from dataclasses import dataclass

from my_enums import SpecialTokens
from sentence_transformers import SentenceTransformer, losses
from transformers import AutoTokenizer, Trainer
import dstc.dstc_utils as dstc_utils


@dataclass
class ContrastiveTokens:
    a_start_token: SpecialTokens
    a_end_token: SpecialTokens
    a_multiple_values: bool
    b_start_token: SpecialTokens
    b_end_token: SpecialTokens
    b_multiple_values: bool
    contrast_with: str


class ContrastiveTrainerHelper:
    tod_tokenizer: AutoTokenizer
    contrastive_model: SentenceTransformer
    token_map: dict[str, int]
    loss_model: None
    max_token_len: int = None
    contrastive_tokens: list[ContrastiveTokens] = None
    is_multitask: bool = False
    ce_loss_weight: float = None
    contrastive_loss_weight: float = None

    def __init__(
        self,
        model_or_path,
        tokenizer,
        max_token_len,
        contrastive_tokens,
        is_multitask,
        ce_loss_weight,
        contrastive_loss_weight,
    ):
        if isinstance(model_or_path, str):
            self.contrastive_model = SentenceTransformer(model_or_path)
        else:
            self.contrastive_model = model_or_path
        self.tod_tokenizer = tokenizer
        special_tokens = [
            SpecialTokens.begin_response,
            SpecialTokens.end_response,
            SpecialTokens.begin_user_action,
            SpecialTokens.end_user_action,
            SpecialTokens.begin_action,
            SpecialTokens.end_action,
            SpecialTokens.prompt_action,
            SpecialTokens.prompt_response,
            SpecialTokens.prompt_dst,
            SpecialTokens.begin_last_user_utterance,
            SpecialTokens.end_last_user_utterance,
        ]
        self.max_token_len = max_token_len

        self.token_map = {}
        for token in special_tokens:
            self.token_map[token] = dstc_utils.get_token_id(tokenizer, token)
        self.loss_model = losses.CosineSimilarityLoss(self.contrastive_model)
        self.contrastive_tokens = contrastive_tokens
        self.is_multitask = is_multitask
        self.ce_loss_weight = ce_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
