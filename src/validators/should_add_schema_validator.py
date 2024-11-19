from omegaconf import DictConfig
from utilities.context_manager import ContextManager
from validators.config_validator import ConfigValidator
import utils


class ShouldAddSchemaValidator(ConfigValidator):
    def validate(self, cfg: DictConfig):
        context_type = cfg.model_type.context_type

        if cfg.should_add_schema:
            if any(
                [
                    ContextManager.is_simple_tod(context_type),
                    ContextManager.is_soloist(context_type),
                ]
            ):
                utils.log(
                    self.logger,
                    message=f"""
                      should_add_schema is set to True for context type {context_type}.
                      Setting this to False. 
                      """,
                    log_prefix="WARNING:",
                )
                cfg.should_add_schema = False
        else:
            if any(
                [
                    ContextManager.is_nlg_strategy(context_type),
                ]
            ):
                utils.log(
                    self.logger,
                    message=f"""
                      should_add_schema is set to False for context type {context_type}.
                      Setting this to True. 
                      """,
                    log_prefix="WARNING:",
                )
                cfg.should_add_schema = True
