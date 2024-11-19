from omegaconf import DictConfig

from utilities.context_manager import ContextManager
from validators.config_validator import ConfigValidator
import utils


class TargetLengthValidator(ConfigValidator):
    def validate(self, cfg: DictConfig):
        context_type = cfg.model_type.context_type
        if ContextManager.is_baseline_api_call(context_type):
            if cfg.test_prompt_max_len > 640:
                utils.log(
                    self.logger,
                    message=f"""
                      You are using a baseline api call strategy with a test_prompt_max_len of {cfg.test_prompt_max_len}.
                      This is incorrect, so setting it to 640.
                      """,
                    log_prefix="WARNING:",
                )
                cfg.test_prompt_max_len = 640
        else:
            if cfg.test_prompt_max_len < 820:
                utils.log(
                    self.logger,
                    message=f"""
                      You are using api call strategy with a test_prompt_max_len of {cfg.test_prompt_max_len}.
                      This is inefficient, so setting it to 820.
                      """,
                    log_prefix="WARNING:",
                )
                cfg.test_prompt_max_len = 820
