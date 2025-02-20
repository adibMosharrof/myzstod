import os
import sys
import hydra
from omegaconf import DictConfig
import torch

sys.path.insert(0, os.path.abspath("./src"))
sys.path.insert(0, os.path.abspath("./"))
from my_enums import Steps


from datamodules.tod_datamodulev2 import TodDataModuleV2
from my_trainers.base_trainer import BaseTrainer
from torch.utils.data import random_split, Subset

from datamodules.tod_dataset import TodDataSet


class InterpretTrainer(BaseTrainer):
    def __init__(self, cfg: dict):
        super().__init__(cfg, dm_class=TodDataModuleV2)

    def get_datasets_from_data_modules(self, dms):
        ds = self.get_dm_dataset(dms[0])
        train = ds["train"]
        dataset_size = len(train)
        train_size = int(0.8 * dataset_size)
        dev_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - dev_size
        i_train, i_dev, i_test = random_split(
            range(dataset_size), [train_size, dev_size, test_size]
        )
        train_dataset = TodDataSet(
            [train.data[i] for i in i_train.indices],
            train.dataset_name,
            train.domain_setting,
            Steps.TRAIN.value,
            train.raw_data_root,
        )
        dev_dataset = TodDataSet(
            [train.data[i] for i in i_dev.indices],
            train.dataset_name,
            train.domain_setting,
            Steps.DEV.value,
            train.raw_data_root,
        )
        test_dataset = TodDataSet(
            [train.data[i] for i in i_test.indices],
            train.dataset_name,
            train.domain_setting,
            Steps.TEST.value,
            train.raw_data_root,
        )
        return train_dataset, dev_dataset, [test_dataset]


@hydra.main(config_path="../../config/interpret/", config_name="interpret_trainer")
def hydra_start(cfg: DictConfig) -> None:
    torch.manual_seed(42)
    itrainer = InterpretTrainer(cfg)
    itrainer.run()


if __name__ == "__main__":
    hydra_start()
