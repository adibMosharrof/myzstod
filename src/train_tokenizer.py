from pathlib import Path
from dotmap import DotMap
import sys
import os
from configs.trainer_config import TrainerConfig

sys.path.insert(0, os.path.abspath("./src"))
from my_enums import Steps
from tod_datamodules import TodDataModule

from tod.turns.zs_tod_turn import TodTurnMultiTaskCsvRow
from tod_datamodules import TodDataModule
from configs.dm_config import DataModuleConfig
import utils


class TrainTokenizer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.out_root = Path("outputs/tokenizer/")
        self.corpus = []

    def get_dms(self):
        steps = Steps.list()
        if self.cfg.is_multi_task:
            return [
                TodDataModule(
                    DataModuleConfig(**self.cfg),
                    steps=steps,
                    tod_turn_row_cls=TodTurnMultiTaskCsvRow,
                    task_name=task_name,
                )
                for task_name in self.cfg.multi_tasks
            ]
        return [
            TodDataModule(
                DataModuleConfig(**self.cfg),
                steps=steps,
            )
        ]

    def create_corpus(self, dm):
        for data in dm:
            corpus_text = "".join([data.context, data.schema, data.target])
            self.corpus.append(corpus_text)

    def train_tokenizer(self):
        tokenizer = utils.get_tokenizer(self.cfg.model_name)
        new_tokenizer = tokenizer.train_new_from_iterator(self.corpus, 52000)
        self.cfg.tokenizer = new_tokenizer
        new_tokenizer.save_pretrained(
            str(self.out_root / self.cfg.custom_tokenizer_name)
        )
        new_tokenizer.push_to_hub(self.cfg.custom_tokenizer_name)

    def run(self) -> None:
        dms = self.get_dms()
        for step in Steps.list():
            for step_dm in dms:
                dm = step_dm.datasets[step]
                if step == Steps.TEST.value:
                    [self.create_corpus(d) for d in dm]
                else:
                    self.create_corpus(dm)

        self.train_tokenizer()


if __name__ == "__main__":
    tt = TrainTokenizer(
        DotMap(
            raw_data_root=Path("data/dstc8-schema-guided-dialogue/"),
            data_prep_out_root="data/processed_data/",
            project_root=Path("/mounts/u-amo-d1/adibm-data/projects/ZSToD"),
            out_root=Path("outputs/tokenizer/"),
            custom_tokenizer_name="adibm/sgd-flan-t5-nlg-tokenizer",
            num_dialogs=[127, 20, 34],
            # num_dialogs=[1, 1, 1],
            data_split_percent=[1, 1, 1],
            model_name="google/flan-t5-large",
            should_add_schema=True,
            should_add_sys_actions=False,
            should_add_user_actions=True,
            should_add_service_results=True,
            train_domain_settings=["seen"],
            dev_domain_settings=["all"],
            test_domain_settings=[["all"]],
        )
    )
    tt.run()
    pass
