from data_prep.data_prep_strategy_factory import DataPrepStrategyFactory
from my_enums import ContextType, DatasetNames
from utilities.context_manager import ContextManager
from validators.config_validator import ConfigValidator
import utils


class DataPrepStrategyValidator(ConfigValidator):
    def validate(self, dataset_name, strategy, context_type):
        if DatasetNames.KETOD.value == dataset_name:
            if not DataPrepStrategyFactory.is_ketod(strategy):
                msg = f"""
                    You are using Ketod data, but context type is {context_type}.
                    Context type should be one of the following {','.join(ContextType.ketod_contexts())}
                    """
                utils.log(self.logger, message=msg, log_prefix="ERROR:")
                raise ValueError(msg)
        elif DatasetNames.BITOD.value == dataset_name:
            if not DataPrepStrategyFactory.is_bitod(strategy):
                msg = f"""
                    You are using Ketod data, but context type is {context_type}.
                    Context type should be one of the following {','.join(ContextType.bitod_contexts())}
                    """
                utils.log(self.logger, message=msg, log_prefix="ERROR:")
                raise ValueError(msg)
        else:
            if any(
                [
                    DataPrepStrategyFactory.is_ketod(strategy),
                    DataPrepStrategyFactory.is_bitod(strategy),
                ]
            ):
                msg = f"""
                    You are using Dstc data, but context type is {context_type}.
                    Context type should be one of the following {','.join(ContextType.dstc_contexts())}
                    """
                utils.log(self.logger, message=msg, log_prefix="ERROR:")
                raise ValueError(msg)
