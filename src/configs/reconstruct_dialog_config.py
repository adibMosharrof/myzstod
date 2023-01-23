import os
from configs.inference_config import InferenceConfig
import utils
from pathlib import Path


class ReconstructDialogConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/data/projects/ZSToD",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        out_dir: str = "reconstruct",
        model_path: str = None,
        predictions_dir: str = "./",
    ):
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = (
            self.project_root / model_path if model_path else Path(predictions_dir)
        )
        self.logger = utils.get_logger()
        files = [
            fname
            for fname in os.listdir(self.predictions_dir)
            if fname.endswith(".csv")
        ]
        if not len(files):
            raise ValueError("No csv files found in the model path")
        self.csv_file_names = files

    @classmethod
    def from_inference_config(
        self, t_config: InferenceConfig
    ) -> "ReconstructDialogConfig":
        return self(
            project_root=t_config.project_root,
            raw_data_root=t_config.raw_data_root,
            predictions_dir=os.getcwd(),
        )
