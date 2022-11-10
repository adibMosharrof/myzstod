import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from contrastive_dataclasses import ContrastiveTokens
from contrastive_datamodule import ContrastiveDataModule
from hydra_configs import ContrastiveConfig, DataModuleConfig
from my_enums import ContrastiveConstants, SpecialTokens, Steps
from sentence_transformers import SentenceTransformer, losses, evaluation, util

from torch.utils.data import DataLoader
import dstc_utils


class Contrastive:
    def __init__(self, cfg: ContrastiveConfig):
        self.cfg = cfg
        self.dm = ContrastiveDataModule(
            DataModuleConfig.from_contrastive_config(self.cfg)
        )

    def run(self) -> SentenceTransformer:
        model = SentenceTransformer(self.cfg.contrastive_model_name)
        if self.cfg.contrastive_model_name == "gpt2":
            model.tokenizer.pad_token = model.tokenizer.eos_token

        word_embedding_model = model._first_module()
        word_embedding_model.tokenizer = dstc_utils.get_tokenizer(
            tokenizer_name=self.cfg.tokenizer_name
        )
        self.dm.contrastive_tokenizer = word_embedding_model.tokenizer
        # model.tokenize = self.dm.contrastive_tokenize
        word_embedding_model.auto_model.resize_token_embeddings(
            len(word_embedding_model.tokenizer)
        )

        contrastive_loss = losses.ContrastiveLoss(model)
        # cosine_loss = losses.CosineSimilarityLoss(model)
        contrastive_tokens = self._get_start_end_tokens()
        eval_data = []
        train_data = []
        for tok in contrastive_tokens:
            eval_data.append(self.dm.get_contrastive_data(tok, Steps.DEV))
            train_data.append(self.dm.get_contrastive_data(tok, Steps.TRAIN))
        eval_data_all = np.concatenate(eval_data, axis=0)
        train_data_all = np.concatenate(train_data, axis=0)

        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            eval_data_all
        )
        train_dl = DataLoader(
            train_data_all,
            batch_size=self.cfg.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
        )

        model.fit(
            # train_objectives=[(train_dl, contrastive_loss), (train_dl, cosine_loss)],
            train_objectives=[(train_dl, contrastive_loss)],
            evaluator=evaluator,
            evaluation_steps=500,
            epochs=self.cfg.train_epochs,
            warmup_steps=100,
            output_path=str(self.cfg.out_dir),
            checkpoint_save_steps=500,
            checkpoint_save_total_limit=2,
            use_amp=True,
            checkpoint_path=str(self.cfg.out_dir),
        )
        return model

    def _get_start_end_tokens(self) -> list[ContrastiveTokens]:
        tokens = []
        for contrast in self.cfg.contrast_with:
            if contrast == ContrastiveConstants.NLG:
                tokens.append(
                    ContrastiveTokens(
                        a_start_token=SpecialTokens.begin_action,
                        a_end_token=SpecialTokens.end_action,
                        a_multiple_values=False,
                        b_start_token=SpecialTokens.begin_response,
                        b_end_token=SpecialTokens.end_response,
                        b_multiple_values=False,
                        contrast_with=contrast,
                    )
                )
            elif contrast == ContrastiveConstants.USER_ACT:
                tokens.append(
                    ContrastiveTokens(
                        b_start_token=SpecialTokens.begin_action,
                        b_end_token=SpecialTokens.end_action,
                        b_multiple_values=False,
                        a_start_token=SpecialTokens.begin_user_action,
                        a_end_token=SpecialTokens.end_user_action,
                        a_multiple_values=False,
                        contrast_with=contrast,
                    )
                )
        return tokens


@hydra.main(config_path="../config/contrastive/", config_name="contrastive")
def hydra_start(cfg: DictConfig) -> None:
    c = Contrastive(ContrastiveConfig(**cfg))
    c.run()


if __name__ == "__main__":
    hydra_start()
