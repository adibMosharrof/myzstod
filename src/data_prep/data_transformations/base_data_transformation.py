from abc import ABC
from utilities.dialog_studio_dataclasses import DsDialog


class BaseDataTransformation(ABC):

    def transform(self, dialog: DsDialog) -> DsDialog:
        raise NotImplementedError("transform method not implemented")
