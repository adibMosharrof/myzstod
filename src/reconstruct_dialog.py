from multiprocessing.sharedctypes import Value
from typing import Optional
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from dstc_dataclasses import DstcAction, DstcDialog, DstcFrame, DstcState, DstcTurn
from hydra_configs import ReconstructDialogConfig
import pandas as pd
from simple_tod_dataclasses import SimpleTodAction, SimpleTodBelief
import utils
from my_enums import SimpleTodConstants, Speaker, SpecialTokens, Steps
import json
from dstc_utils import get_text_in_between
import humps
from itertools import cycle


class ReconstructDialog:
    def __init__(self, cfg: ReconstructDialogConfig):
        self.cfg = cfg

    def _read_dialog_from_file(self, dialog_id: str) -> Optional[DstcDialog]:
        id, _ = dialog_id.split("_")
        id_str = (3 - len(id)) * "0" + id
        path = self.cfg.raw_data_root / Steps.TEST.value / f"dialogues_{id_str}.json"
        json_data = utils.read_json(path)
        for data in json_data:
            if data["dialogue_id"] == dialog_id:
                dialog = DstcDialog.from_dict(data)
                return dialog
        return None

    def _get_sys_actions(self, target: str) -> list[DstcAction]:
        actions_str = get_text_in_between(
            target,
            SpecialTokens.begin_action,
            SpecialTokens.end_action,
            default_value="",
        )
        actions: list[DstcAction] = []
        for action_str in actions_str.split(SimpleTodConstants.ITEM_SEPARATOR):
            tod_action = SimpleTodAction.from_string(action_str)
            dstc_action = DstcAction(
                act=tod_action.action_type,
                slot=tod_action.slot_name,
                canonical_values=[],
                values=[],
            )
            actions.append(dstc_action)
        return actions

    def _get_user_state(self, dst: str) -> DstcState:
        active_intent = get_text_in_between(
            dst, SpecialTokens.begin_intent, SpecialTokens.end_intent, default_value=""
        )
        requested_slots_txt = get_text_in_between(
            dst,
            SpecialTokens.begin_requested_slots,
            SpecialTokens.end_requested_slots,
            default_value="",
        )
        requested_slots = []
        if requested_slots_txt:
            requested_slots = requested_slots_txt.split(
                SimpleTodConstants.ITEM_SEPARATOR
            )

        beliefs_str = get_text_in_between(
            dst, SpecialTokens.begin_belief, SpecialTokens.end_belief, default_value=""
        )

        slot_values = {}
        if beliefs_str:
            for belief_str in beliefs_str.split(SimpleTodConstants.ITEM_SEPARATOR):
                tod_belief = SimpleTodBelief.from_string(belief_str)
                # slot_name = humps.depascalize(tod_belief.slot_name)
                slot_name = tod_belief.slot_name
                slot_values[slot_name] = tod_belief.values
        return DstcState(
            active_intent=active_intent,
            slot_values=slot_values,
            requested_slots=requested_slots,
        )

    def _get_user_frames(self, text: str, gt_user_turn: DstcTurn) -> list[DstcFrame]:
        dsts = get_text_in_between(
            text,
            SpecialTokens.begin_dst,
            SpecialTokens.end_dst,
            multiple_values=True,
            default_value=[],
        )
        if len(gt_user_turn.frames) > len(dsts):
            a = 1
        frames = []
        for i, gt_frame in enumerate(gt_user_turn.frames):
            # for dst, gt_frame in zip(cycle(dsts), gt_user_turn.frames):
            try:
                dst = dsts[i]
            except IndexError:
                dst = ""
            frame = DstcFrame(
                state=self._get_user_state(dst),
                actions=[],
                service=gt_frame.service,
                slots=[],
            )
            frames.append(frame)
        return frames
        return [
            DstcFrame(
                state=self._get_user_state(dst),
                actions=[],
                service=gt_frame.service,
                slots=[],
            )
            for dst, gt_frame in zip(cycle(dsts), gt_user_turn.frames)
        ]

    def _extract_turn(
        self, pred: str, context: str, gt_user_turn: DstcTurn, gt_sys_turn: DstcTurn
    ) -> list[DstcTurn]:

        user_utterance = get_text_in_between(
            context,
            SpecialTokens.begin_last_user_utterance,
            SpecialTokens.end_last_user_utterance,
            default_value="",
        )

        user_frames = self._get_user_frames(pred, gt_user_turn)
        user_turn = DstcTurn(
            speaker=Speaker.USER.value, frames=user_frames, utterance=user_utterance
        )

        utterance = get_text_in_between(
            pred,
            SpecialTokens.begin_response,
            SpecialTokens.end_response,
            default_value="",
        )
        sys_frame = DstcFrame(
            actions=[], slots=[], service=gt_sys_turn.frames[0].service
        )
        sys_turn = DstcTurn(
            speaker=Speaker.SYSTEM.value, utterance=utterance, frames=[]
        )
        sys_frame.actions.append(self._get_sys_actions(pred))
        sys_turn.frames.append(sys_frame)

        return [user_turn, sys_turn]

    def run(self):
        for f_name in self.cfg.csv_file_names:
            # df = pd.read_csv(self.cfg.csv_file_names, keep_default_na=False)
            df = pd.read_csv(self.cfg.model_path / f_name, keep_default_na=False)
            df_dialogs = df.groupby("dialog_id").agg(list)
            out_dict = {}
            for id, dialog in tqdm(df_dialogs.iterrows()):
                # utils.write_json(dialog_json, self.cfg.out_dir / f"{id}.json")
                gt_dialog = self._read_dialog_from_file(id)
                dstc_dialog = DstcDialog(
                    dialogue_id=id, turns=[], services=gt_dialog.services
                )
                for turn_id, target, pred, context, (gt_user_turn, gt_sys_turn) in zip(
                    dialog.turn_id,
                    dialog.target,
                    dialog.prediction,
                    dialog.context,
                    utils.grouper(gt_dialog.turns, 2),
                ):
                    user_turn, sys_turn = self._extract_turn(
                        pred, context, gt_user_turn, gt_sys_turn
                    )
                    dstc_dialog.turns.append(user_turn)
                    dstc_dialog.turns.append(sys_turn)
                out_dict[id] = dstc_dialog.to_dict()

                utils.write_json(out_dict, self.cfg.out_dir / f"{id}.json")


@hydra.main(config_path="../config/reconstruct/", config_name="reconstruct")
def hydra_start(cfg: DictConfig) -> None:
    stt = ReconstructDialog(ReconstructDialogConfig(**cfg))
    stt.run()


if __name__ == "__main__":
    hydra_start()
