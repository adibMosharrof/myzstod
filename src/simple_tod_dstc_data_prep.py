import copy
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional
import hydra
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm
import humps

import utils

from dstc_dataclasses import DstcDialog, DstcFrame, DstcSchema, DstcTurn, Steps
from dstc_utils import get_csv_data_path, get_dialog_file_paths, get_dstc_service_name


from simple_tod_dataclasses import (
    SimpleTodAction,
    SimpleTodBelief,
    SimpleTodConstants,
    SimpleTodContext,
    SimpleTodTarget,
    SimpleTodTurn,
)


class SimpleTODDSTCDataPrep:
    def __init__(
        self,
        project_root: str,
        data_root: str,
        out_root: str,
        num_dialogs: List[int] = None,
        delexicalize: bool = True,
        overwrite: List[bool] = None,
        domains: List[str] = None,
        num_turns: int = 55,
    ):
        self.project_root = Path(project_root)
        self.data_root = self.project_root / data_root
        self.out_root = self.project_root / out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.num_dialogs = num_dialogs
        # self.services = self._get_seen_services(services, self.data_root)
        self.delexicalize = delexicalize
        self.overwrite = overwrite or [False, False, False]
        self.domains = domains or ["Buses", "Hotels", "Events"]
        self.num_turns = num_turns

    def _prepare_context(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        prev_tod_turn: SimpleTodTurn,
        schemas: Dict[str, DstcSchema],
    ):
        if not prev_tod_turn:
            context = SimpleTodContext()
        else:
            context = copy.deepcopy(prev_tod_turn.context)
            if len(context.system_utterances) == self.num_turns:
                context.system_utterances.popleft()
            context.system_utterances.append(
                prev_tod_turn.context.next_system_utterance
            )

        if user_turn:
            utterance = user_turn.utterance
            if self.delexicalize:
                utterance = self._delexicalize_utterance(user_turn, schemas)
            if len(context.user_utterances) == self.num_turns:
                context.user_utterances.popleft()
            context.user_utterances.append(utterance)
        if system_turn:
            utterance = system_turn.utterance
            if self.delexicalize:
                utterance = self._delexicalize_utterance(system_turn, schemas)
            context.next_system_utterance = utterance
        return context

    def _prepare_belief(self, user_turn: DstcTurn) -> List[SimpleTodBelief]:
        beliefs = []
        for frame in user_turn.frames:
            if not frame.state:
                continue
            for slot_name, value in frame.state.slot_values.items():
                beliefs.append(
                    SimpleTodBelief(
                        get_dstc_service_name(frame.service),
                        humps.camelize(slot_name),
                        " ".join(value),
                    )
                )
        return beliefs

    def _create_user_action(self, actions, frames: List[DstcFrame]):
        for frame in frames:
            for action in frame.actions:
                actions.append(
                    SimpleTodAction(frame.service, action.act, " ".join(action.values))
                )

    def _create_system_action(self, actions, frames: List[DstcFrame]):
        for frame in frames:
            for action in frame.actions:
                actions.append(
                    SimpleTodAction(
                        get_dstc_service_name(frame.service),
                        action.act,
                        humps.camelize(action.slot),
                    )
                )

    def _prepare_action(self, system_turn: DstcTurn) -> List[SimpleTodAction]:
        actions = []
        if system_turn:
            self._create_system_action(actions, system_turn.frames)
        return actions

    def _delexicalize_utterance(
        self, turn: DstcTurn, schemas: Dict[str, DstcSchema]
    ) -> str:
        delexicalized_utterance = turn.utterance
        for frame in turn.frames:
            schema = schemas[frame.service]
            for action in frame.actions:
                for value in action.values:
                    slot = next(
                        (slot for slot in schema.slots if slot.name == action.slot),
                        None,
                    )
                    if not slot:
                        continue
                    replacement = f"<{get_dstc_service_name(frame.service)}_{humps.camelize(action.slot)}>"
                    delexicalized_utterance = delexicalized_utterance.replace(
                        value, replacement
                    )
        return delexicalized_utterance

    def _prepare_response(
        self, system_turn: DstcTurn, schemas: Dict[str, DstcSchema]
    ) -> str:
        if not system_turn:
            return None
        if not self.delexicalize:
            return system_turn.utterance
        return self._delexicalize_utterance(system_turn, schemas)

    def _prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: Dict[str, DstcSchema],
    ):
        beliefs = self._prepare_belief(user_turn)
        actions = self._prepare_action(system_turn)
        response = self._prepare_response(system_turn, schemas)
        active_intent = user_turn.get_active_intent()
        requested_slots = user_turn.get_requested_slots()
        return SimpleTodTarget(
            beliefs, actions, response, active_intent, requested_slots
        )

    def _prepare_turn(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        prev_tod_turn: SimpleTodTurn,
        schemas: Dict[str, DstcSchema],
    ) -> SimpleTodTurn:
        target = None
        context = self._prepare_context(user_turn, system_turn, prev_tod_turn, schemas)
        target = self._prepare_target(user_turn, system_turn, schemas)
        return SimpleTodTurn(context, target)

    def _is_dialogue_in_domain(self, dialogue_services: List[str]) -> bool:
        return all(
            get_dstc_service_name(ds) in self.domains for ds in dialogue_services
        )

    def _prepare_dialog(
        self, dstc_dialog: DstcDialog, schemas: Dict[str, DstcSchema]
    ) -> Optional[List[SimpleTodTurn]]:
        tod_turns = []
        tod_turn = None
        if not self._is_dialogue_in_domain(dstc_dialog.services):
            return None

        for i, (user_turn, system_turn) in enumerate(
            utils.grouper(dstc_dialog.turns, 2)
        ):
            tod_turn = self._prepare_turn(user_turn, system_turn, tod_turn, schemas)
            tod_turn.dialog_id = dstc_dialog.dialogue_id
            tod_turn.turn_id = i + 1
            tod_turns.append(tod_turn.to_csv_row())
        return tod_turns

    def _prepare_dialog_file(
        self, path: Path, schemas: Dict[str, DstcSchema]
    ) -> np.ndarray:
        data = []
        dialog_json_data = utils.read_json(path)
        for d in dialog_json_data:
            dialog = DstcDialog.from_json(json.dumps(d))
            prepped_dialog = self._prepare_dialog(dialog, schemas)
            if prepped_dialog is None:
                continue
            data.append(prepped_dialog)
        if not len(data):
            return np.array(data)
        return np.concatenate(data, axis=0)

    def _get_schemas(self, step: str) -> Dict[str, DstcSchema]:
        path = self.data_root / step / "schema.json"
        schema_json = utils.read_json(path)
        schemas = {}
        for s in schema_json:
            schema = DstcSchema.from_json(json.dumps(s))
            schema.step = step
            schemas[schema.service_name] = schema
        return schemas

    def run(self):
        steps = Steps.list()
        for step, num_dialog, should_overwrite in tqdm(
            zip(steps, self.num_dialogs, self.overwrite)
        ):
            step_dir = Path(self.out_root / step)
            step_dir.mkdir(parents=True, exist_ok=True)
            dialog_paths = get_dialog_file_paths(self.data_root, step)
            schemas = self._get_schemas(step)
            out_data = []
            if num_dialog == "None":
                num_dialog = len(dialog_paths)
            csv_file_path = get_csv_data_path(
                step=step,
                num_dialogs=num_dialog,
                delexicalized=self.delexicalize,
                domains=self.domains,
                processed_data_root=self.out_root,
                num_turns=self.num_turns,
            )
            if csv_file_path.exists() and not should_overwrite:
                print(f"{step} csv file already exists, so skipping")
                continue
            for dp in tqdm(dialog_paths[:num_dialog]):
                dialog_data = self._prepare_dialog_file(dp, schemas)
                if not len(dialog_data):
                    continue
                out_data.append(dialog_data)
            headers = ["dialog_id", "turn_id", "context", "target"]
            if len(out_data) == 0:
                print(f"No data for {step}")
                continue
            csv_data = np.concatenate(out_data, axis=0)
            utils.write_csv(headers, csv_data, csv_file_path)


@hydra.main(config_path="../config/data_prep/", config_name="simple_tod")
def hydra_start(cfg: DictConfig) -> None:
    stdp = SimpleTODDSTCDataPrep(
        cfg.project_root,
        cfg.data_root,
        cfg.out_root,
        num_dialogs=cfg.num_dialogs,
        delexicalize=cfg.delexicalize,
        overwrite=cfg.overwrite,
        domains=cfg.domains,
        num_turns=cfg.num_turns,
    )
    stdp.run()


if __name__ == "__main__":
    hydra_start()
