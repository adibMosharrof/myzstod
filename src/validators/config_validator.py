from abc import ABC, abstractmethod
from omegaconf import DictConfig
import logging


class ConfigValidator(ABC):

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger()

    @abstractmethod
    def validate(self, config: DictConfig):
        """
        Perform validation on the given config.
        Should raise an exception if validation fails.
        """
        pass
