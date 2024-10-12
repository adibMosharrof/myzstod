import copy
from datamodules.tod_dataset import TodDataSet
from typing import Union
from omegaconf import ListConfig
from base_datamodule import SimpleTodDataSet
from data_prep.data_prep_class_factory import DataPrepClassFactory
from datamodules.data_filters.base_data_filter import BaseDataFilter
from datamodules.data_filters.split_percent_filter import SplitPercentFilter
from datamodules.dm_dataclasses import StepData
from my_enums import Steps
from tod.turns.zs_tod_turn import TodTurnCsvRow
from configs.dm_config import DataModuleConfig
from tod.turns.zs_tod_turn import TodTurnCsvRow
import utils


class TodDataModuleV2:

    _huggingface_ignore_label_id = -100

    def __init__(
        self,
        cfg: DataModuleConfig,
        steps: list[Steps] = None,
        tod_turn_row_cls=TodTurnCsvRow,
        data_filters: list[BaseDataFilter] = None,
        data_augmentations=None,
    ):
        self.cfg = cfg
        self.tod_turn_row_cls = tod_turn_row_cls
        self.steps = steps or Steps.list()
        self.data_filters = data_filters or []
        self.datasets: dict[str, SimpleTodDataSet] = {}
        self.data_augmentations = data_augmentations or []

    def setup(self):
        for step in self.steps:
            step_data = self.get_step_data(step)
            if isinstance(step_data.domain_settings[0], (list, ListConfig)):
                self.datasets[step] = []
                for domain_setting in step_data.domain_settings:
                    ds = self.setup_single_run(step, step_data, domain_setting)
                    if len(ds):
                        self.datasets[step].append(ds)
            else:
                self.datasets[step] = self.setup_single_run(
                    step,
                    step_data,
                    step_data.domain_settings,
                )

    def setup_single_run(
        self, step: str, step_data: StepData, domain_setting: Union[str, list[str]]
    ) -> TodDataSet:
        cfg = copy.deepcopy(self.cfg)
        cfg.step_name = step_data.name
        cfg.num_dialogs = step_data.num_dialog
        cfg.overwrite = step_data.overwrite
        cfg.domain_setting = domain_setting

        data_prep_instance = DataPrepClassFactory.create_data_prep_instance(cfg)
        self.prepare_data(data_prep_instance)
        csv_path = utils.get_csv_data_path(
            step,
            step_data.num_dialog,
            cfg=data_prep_instance.cfg,
        )
        try:
            data = utils.read_csv_dataclass(csv_path, self.tod_turn_row_cls)
        except FileNotFoundError:
            data = []

        for filter in self.data_filters:
            data = filter.apply(data)
        split_filter = SplitPercentFilter(percent=step_data.split_percent)
        data = split_filter.apply(data)
        return TodDataSet(
            data=data,
            dataset_name=cfg.dataset_name,
            step_name=step,
            domain_setting=domain_setting,
            raw_data_root=cfg.raw_data_root,
        )

    def prepare_data(self, data_prep_instance: any):
        if self.cfg.accelerator.is_main_process:
            data_prep_instance.run()
        self.cfg.accelerator.wait_for_everyone()

    def get_step_data(self, step: Steps) -> StepData:
        index = Steps.get_index(step)
        return StepData(
            step,
            self.cfg.num_dialogs[index],
            self.cfg.overwrite[index],
            self.cfg.data_split_percent[index],
            getattr(self.cfg, f"{step}_domain_settings"),
        )
