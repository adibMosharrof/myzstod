from data_prep.bitod.bitod_strategy import BitodStrategy
from data_prep.ketod.ketod_nlg_api_call_strategy import KetodNlgApiCallStrategy
from my_enums import ContextType, DatasetNames
from pathlib import Path
from configs.dataprep_config import DataPrepConfig
from configs.dm_config import DataModuleConfig
from data_prep.bitod.bitod_data_prep import BitodDataPrep
from data_prep.data_prep_strategy_factory import DataPrepStrategyFactory
from data_prep.dstc_base_data_prep import DstcBaseDataPrep
from data_prep.ketod.ketod_base_data_prep import KetodBaseDataPrep
from schema.schema_factory import SchemaFactory
from validators.data_prep_strategy_validator import DataPrepStrategyValidator


class DataPrepClassFactory:
    @staticmethod
    def create_data_prep_instance(cfg: DataModuleConfig, schemas):
        if isinstance(cfg.raw_data_root, str):
            cfg.raw_data_root = Path(cfg.raw_data_root)
        dp_cfg = DataPrepConfig.from_dm_config(cfg)
        strategy = DataPrepStrategyFactory.get_strategy(dp_cfg, dp_cfg.context_type)
        schema_loader = SchemaFactory.create_schema_loader(dp_cfg.context_type)

        strategy_validator = DataPrepStrategyValidator()
        strategy_validator.validate(cfg.dataset_name, strategy, dp_cfg.context_type)
        data_prep_instance = DataPrepClassFactory.get_data_prep_instance(
            dp_cfg, cfg.dataset_name, strategy, schema_loader, schemas
        )
        return data_prep_instance
        if "ketod" in cfg.dataset_name:
            if not isinstance(strategy, KetodNlgApiCallStrategy):
                raise ValueError(
                    f"You are using Ketod data, but context type is {dp_cfg.context_type}."
                    f"Context type should be one of the following {','.join(ContextType.ketod_contexts())}"
                )
            return KetodBaseDataPrep(dp_cfg, strategy)
        elif "bitod" in cfg.dataset_name:
            if not isinstance(strategy, BitodStrategy):
                raise ValueError(
                    f"You are using Ketod data, but context type is {dp_cfg.context_type}."
                    f"Context type should be one of the following {','.join(ContextType.bitod_contexts())}"
                )
            return BitodDataPrep(dp_cfg, strategy)
        else:
            if isinstance(strategy, (KetodNlgApiCallStrategy, BitodStrategy)):
                raise ValueError(
                    f"You are using Dstc data, but context type is {dp_cfg.context_type}."
                    f"Context type should be one of the following {','.join(ContextType.dstc_contexts())}"
                )
            return DstcBaseDataPrep(
                dp_cfg, strategy, schema_loader=schema_loader, schemas=schemas
            )

    @staticmethod
    def get_data_prep_instance(
        dp_cfg, dataset_name: str, strategy, schema_loader, schemas
    ):
        if DatasetNames.KETOD.value == dataset_name:
            return KetodBaseDataPrep(dp_cfg, strategy)
        elif DatasetNames.BITOD.value == dataset_name:
            return BitodDataPrep(dp_cfg, strategy)
        else:
            return DstcBaseDataPrep(dp_cfg, strategy, schema_loader, schemas)
