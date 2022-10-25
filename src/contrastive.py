import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from contrastive_datamodule import ContrastiveDataModule
from hydra_configs import ConstrastiveConfig, DataModuleConfig
from my_enums import ContrastiveConstrants, SpecialTokens, Steps
from sentence_transformers import SentenceTransformer, losses, evaluation, util

from torch.utils.data import DataLoader
import utils


class Contrastive:
    def __init__(self, cfg: ConstrastiveConfig):
        self.cfg = cfg
        self.dm = ContrastiveDataModule(
            DataModuleConfig.from_contrastive_config(self.cfg)
        )

    def run(self):

        model = SentenceTransformer(self.cfg.model_name)
        contrastive_loss = losses.ContrastiveLoss(model)
        cosine_loss = losses.CosineSimilarityLoss(model)
        start_token, end_token = self._get_start_end_tokens()
        evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
            self.dm.get_contrastive_data(start_token, end_token, Steps.DEV)
        )
        train_dl = DataLoader(
            self.dm.get_contrastive_data(start_token, end_token),
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

    def test(self):
        start_token, end_token = self._get_start_end_tokens()
        test_dl = self.dm.get_contrastive_data(start_token, end_token, Steps.DEV)
        model = SentenceTransformer(str(self.cfg.model))
        out = model(test_dl[0])
        a = 1

    def _get_start_end_tokens(self):
        if self.cfg.contrast_with == ContrastiveConstrants.NLG:
            start_token = SpecialTokens.begin_response
            end_token = SpecialTokens.end_response
        elif self.cfg.contrast_with == ContrastiveConstrants.USER_ACT:
            start_token = SpecialTokens.begin_user_action
            end_token = SpecialTokens.end_user_action
        return start_token, end_token


@hydra.main(config_path="../config/contrastive/", config_name="contrastive")
def hydra_start(cfg: DictConfig) -> None:
    c = Contrastive(ConstrastiveConfig(**cfg))
    if c.cfg.model:
        c.test()
    else:
        c.run()
    # c.run()


if __name__ == "__main__":
    hydra_start()
