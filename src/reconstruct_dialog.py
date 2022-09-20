import hydra
from omegaconf import DictConfig
from dstc_dataclasses import DstcDialog, DstcTurn
from hydra_configs import ReconstructDialogConfig
import pandas as pd
import utils
from my_enums import Steps
import json


class ReconstructDialog:
    def __init__(self, cfg: ReconstructDialogConfig):
        self.cfg = cfg

    def read_dialog_from_file(self, dialog_id: str):
        path = self.cfg.raw_data_root / Steps.TEST / f"dialogues_{dialog_id}.json"
        json_data = utils.read_json(path)
        return json_data
        return DstcDialog.from_json(json.dumps(json_data))

    def run(self):
        df = pd.read_csv(self.cfg.predictions_csv_path)
        df_dialogs = df.groupby("dialog_id").agg(list)
        for id, dialog in df_dialogs.iterrows():
            dialog_json = self.read_dialog_from_file(id)
            # dstc_dialog = DstcDialog()
            # for turn_id, target, pred in zip(
            #     dialog.turn_id, dialog.target, dialog.prediction
            # ):
            #     dstc_turn = DstcTurn()


@hydra.main(config_path="../config/reconstruct/", config_name="reconstruct")
def hydra_start(cfg: DictConfig) -> None:
    stt = ReconstructDialog(ReconstructDialogConfig(**cfg))
    stt.run()


if __name__ == "__main__":
    hydra_start()
