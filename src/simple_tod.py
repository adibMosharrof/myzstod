from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import glob
from tqdm import tqdm
import utils
from my_dataclasses import DstcFrame, Speaker, TodTurn, TodContext, DstcDialog, DstcTurn
import copy
import json
import numpy as np


class SimpleTODDataPrep:
    def __init__(self, data_root: str, out_root: str):
        self.data_root = Path(data_root)
        self.out_root = self.data_root / out_root
        self.out_root.mkdir(parents=True, exist_ok=True)

    def _get_dialog_file_paths(self, step):
        file_paths = glob.glob(str(self.data_root / step / "dialogues_*"))
        return file_paths

    def _prepare_context(
        self, user_turn: DstcTurn, system_turn: DstcTurn, prev_tod_turn: TodTurn
    ):
        a = 1
        context = (
            TodContext() if not prev_tod_turn else copy.deepcopy(prev_tod_turn.context)
        )
        if user_turn:
            context.user_utterances.append(user_turn.utterance)
        if system_turn:
            context.system_utterances.append(system_turn.utterance)
        return context

    def _prepare_target(
        self,
        dstc_turn: DstcTurn,
        prev_tod_turn: TodTurn,
    ):
        a = 1

    def _prepare_turn(
        self, user_turn: DstcTurn, system_turn: DstcTurn, prev_tod_turn: TodTurn
    ) -> TodTurn:
        target = None
        context = self._prepare_context(user_turn, system_turn, prev_tod_turn)
        # target = self._prepare_target(dstc_turn, prev_tod_turn)

        return TodTurn(context, target)

    def _prepare_dialog(self, dstc_dialog: DstcDialog):
        tod_turns = []
        tod_turn = None
        for system_turn, user_turn in utils.grouper(dstc_dialog.turns, 2):
            tod_turn = self._prepare_turn(user_turn, system_turn, tod_turn)
            tod_turns.append(tod_turn)
        return tod_turns

    def _prepare_dialog_file(self, path):
        data = []
        dialog_json_data = utils.read_json(path)
        for d in dialog_json_data:
            dialog = DstcDialog.from_json(json.dumps(d))
            prepped_dialog = self._prepare_dialog(dialog)
            data = np.concatenate([data, prepped_dialog])
        return data

    def run(self):
        steps = ["train", "dev", "test"]
        for step in tqdm(steps):
            step_dir = Path(self.out_root / step)
            step_dir.mkdir(parents=True, exist_ok=True)
            dialog_paths = self._get_dialog_file_paths(step)
            out_data = []
            for dp in dialog_paths:
                dialog_data = self._prepare_dialog_file(dp)
                out_data = np.concatenate([out_data, dialog_data])
        a = 1


@hydra.main(config_path="../config/data_prep/", config_name="simple_tod")
def hydra_start(cfg: DictConfig) -> None:
    stdp = SimpleTODDataPrep(
        cfg.data_root,
        cfg.out_root,
        # cfg.metrics,
        # log_dir=cfg.log_dir,
        # title=cfg.plot_title,
    )
    stdp.run()


if __name__ == "__main__":
    hydra_start()
