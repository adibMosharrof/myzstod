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
            target, SpecialTokens.begin_action, SpecialTokens.end_action
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

    def _get_user_state(self, target: str) -> DstcState:
        active_intent = get_text_in_between(
            target, SpecialTokens.begin_intent, SpecialTokens.end_intent
        )

        requested_slots_txt = get_text_in_between(
            target,
            SpecialTokens.begin_requested_slots,
            SpecialTokens.end_requested_slots,
        )
        requested_slots = []
        if requested_slots_txt:
            requested_slots = requested_slots_txt.split(
                SimpleTodConstants.ITEM_SEPARATOR
            )

        beliefs_str = get_text_in_between(
            target, SpecialTokens.begin_belief, SpecialTokens.end_belief
        )
        slot_values = {}
        for belief_str in beliefs_str.split(SimpleTodConstants.ITEM_SEPARATOR):
            tod_belief = SimpleTodBelief.from_string(belief_str)
            slot_values[tod_belief.slot_name] = [tod_belief.value]
        return DstcState(
            active_intent=active_intent,
            slot_values=slot_values,
            requested_slots=requested_slots,
        )

    def _extract_turn(
        self, pred: str, context: str, gt_user_turn: DstcTurn, gt_sys_turn: DstcTurn
    ) -> list[DstcTurn]:

        user_utterance = get_text_in_between(
            context,
            SpecialTokens.begin_last_user_utterance,
            SpecialTokens.end_last_user_utterance,
        )
        user_frame = DstcFrame(
            state=self._get_user_state(pred),
            actions=[],
            slots=[],
            service=gt_user_turn.frames[0].service,
        )
        user_frame.service = gt_user_turn.frames[0].full_service

        user_turn = DstcTurn(
            speaker=Speaker.USER.value, frames=[], utterance=user_utterance
        )
        user_turn.frames.append(user_frame)

        utterance = get_text_in_between(
            pred, SpecialTokens.begin_response, SpecialTokens.end_response
        )
        sys_frame = DstcFrame(
            actions=[], slots=[], service=gt_sys_turn.frames[0].service
        )
        sys_frame.service = gt_sys_turn.frames[0].full_service
        sys_turn = DstcTurn(
            speaker=Speaker.SYSTEM.value, utterance=utterance, frames=[]
        )
        sys_frame.actions.append(self._get_sys_actions(pred))
        sys_turn.frames.append(sys_frame)

        return [user_turn, sys_turn]

    def run(self):
        df = pd.read_csv(self.cfg.predictions_csv_path)
        df_dialogs = df.groupby("dialog_id").agg(list)
        out_dict = {}
        for id, dialog in tqdm(df_dialogs.iterrows()):
            # utils.write_json(dialog_json, self.cfg.out_dir / f"{id}.json")
            gt_dialog = self._read_dialog_from_file(id)
            dstc_dialog = DstcDialog(
                dialogue_id=id, turns=[], services=gt_dialog.full_services
            )
            dstc_dialog.services = gt_dialog.full_services
            for turn_id, target, pred, context, (gt_user_turn, gt_sys_turn) in zip(
                dialog.turn_id,
                dialog.target,
                dialog.prediction,
                dialog.context,
                utils.grouper(gt_dialog.turns, 2),
            ):
                user_turn, sys_turn = self._extract_turn(
                    target, context, gt_user_turn, gt_sys_turn
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
