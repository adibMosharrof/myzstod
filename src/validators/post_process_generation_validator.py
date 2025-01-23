from omegaconf import DictConfig

from utilities.context_manager import ContextManager
from validators.config_validator import ConfigValidator
import utils


class PostProcessGenerationValidator(ConfigValidator):
    def validate(self, cfg: DictConfig):
        context_type = cfg.model_type.context_type
        if any(
            [
                ContextManager.is_zs_simple_tod_api_call(context_type),
                ContextManager.is_soloist(context_type),
            ]
        ):
            should_post_process = cfg.get("should_post_process", None)
            if not should_post_process:
                utils.log(
                    self.logger,
                    message=f"""
                      You are using a baseline api call strategy with should post process set to {should_post_process}.
                      This is incorrect, so setting it to True.
                      """,
                    log_prefix="WARNING:",
                )
                cfg.should_post_process = True
