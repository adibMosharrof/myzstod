import utils
from validators.config_validator import ConfigValidator


class ShouldTrainValidator(ConfigValidator):
    def validate(self, cfg):
        if cfg.should_train:
            return
        if not self.cfg.model_type.model_path:
            msg = """
                model_type.model_path is required since should_train is set to False.
                Try setting should_train to True or provide a model_path
                """
            utils.log(self.logger, message=msg, log_prefix="ERROR:")
            raise ValueError(msg)
