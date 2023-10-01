import copy
import itertools
from pathlib import Path
import sys
import os
import numpy as np

from tqdm import tqdm

sys.path.insert(0, os.path.abspath("./src"))

from tod.zs_tod_target import ZsTodTarget
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief
from tod.zs_tod_dst import ZsTodDst

from tod.turns.zs_tod_turn import ZsTodTurn
from tod.turns.turn_csv_row import TurnCsvRowBase
from tod.turns.turn_csv_row_factory import TurnCsvRowFactory
from tod.zs_tod_context import ZsTodContext


import hydra
from omegaconf import DictConfig
from multi_woz.multi_woz_dialog_act import MultiWozAct, MultiWozDialogAct
from configs.dataprep_config import DataPrepConfig
from configs.multi_woz_data_prep_config import MultiWozDataPrepConfig
from multi_woz.multi_woz_schema import MultiWozSchema
from my_enums import Steps, ZsTodConstants
import utils

from pathos.multiprocessing import ProcessingPool as Pool
from sgd_dstc8_data_model.dstc_dataclasses import DstcDialog, DstcTurn


class TodMultiWoz22DataPrep:
    def __init__(self, cfg: MultiWozDataPrepConfig):
        self.cfg = cfg

    def _is_dialogue_in_domain(self, dialogue_services: list[str]) -> bool:
        return all(ds in self.cfg.domains for ds in dialogue_services)

    def get_schemas(self) -> dict[str, MultiWozSchema]:
        file_path = self.cfg.raw_data_root / "schema.json"
        schema_json = utils.read_json(file_path)
        out = {}
        for s in schema_json:
            schema = MultiWozSchema.from_dict(s)
            out[schema.service_name] = schema
        return out

    def _read_act(self, dial_id: str, act_dict: dict[str, any]) -> MultiWozDialogAct:
        out = MultiWozDialogAct(dialogue_id=dial_id, turns={})
        for turn_id, turn in act_dict.items():
            for key, value in turn.items():
                if key == "span_info":
                    continue
                actions = []
                for act_name, act_values in value.items():
                    for val in act_values:
                        actions.append(
                            MultiWozAct(
                                action_type=act_name, slot_name=val[0], value=val[1]
                            )
                        )
            out.turns[turn_id] = actions
        return out

    def get_acts(self) -> dict[str, MultiWozDialogAct]:
        file_path = self.cfg.raw_data_root / "dialog_acts.json"
        acts_json = utils.read_json(file_path)
        woz_acts = {}
        for dialog_id, acts in acts_json.items():
            row = self._read_act(dialog_id, acts)
            woz_acts[dialog_id] = row
        return woz_acts

    def _prepare_dst(self, user_turn: DstcTurn) -> list[ZsTodBelief]:
        dsts = []
        for frame in user_turn.frames:
            if not frame.state or frame.state.active_intent == "NONE":
                continue
            beliefs = []
            active_intent = frame.state.active_intent
            requested_slots = [
                "".join(
                    [
                        frame.short_service,
                        ZsTodConstants.DOMAIN_SLOT_SEPARATOR,
                        slot,
                    ]
                )
                for slot in frame.state.requested_slots
            ]
            for slot_name, value in frame.state.slot_values.items():
                beliefs.append(
                    ZsTodBelief(
                        frame.short_service,
                        # humps.camelize(slot_name),
                        slot_name,
                        value,
                    )
                )
            dsts.append(ZsTodDst(beliefs, active_intent, requested_slots))
        return dsts

    def _get_actions(self, woz_acts: MultiWozAct) -> list[ZsTodAction]:
        actions = []
        for act in woz_acts:
            domain, action_type = act.action_type.split("-")
            actions.append(
                ZsTodAction(
                    domain,
                    action_type,
                    act.slot_name,
                    act.value,
                )
            )
        return actions

    def _prepare_response(
        self, system_turn: DstcTurn, schemas: dict[str, MultiWozSchema]
    ) -> str:
        if not system_turn:
            return None
        if not self.cfg.delexicalize:
            return system_turn.utterance
        return self._delexicalize_utterance(system_turn, schemas)

    def _prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, MultiWozSchema],
        woz_user_acts: list[MultiWozAct],
        woz_sys_acts: list[MultiWozAct],
    ) -> ZsTodTarget:
        dsts = self._prepare_dst(user_turn)
        actions = self._get_actions(woz_sys_acts)
        user_actions = self._get_actions(woz_user_acts)
        response = self._prepare_response(system_turn, schemas)
        return ZsTodTarget(
            dsts=dsts, actions=actions, user_actions=user_actions, response=response
        )

    def _prepare_context(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        prev_tod_turn: ZsTodTurn,
        schemas: dict[str, MultiWozSchema],
        woz_user_acts: list[MultiWozAct],
        woz_sys_acts: list[MultiWozAct],
    ):
        if not prev_tod_turn:
            context = ZsTodContext(max_length=self.cfg.num_turns)
            context.should_add_sys_actions = self.cfg.should_add_sys_actions
        else:
            context = copy.deepcopy(prev_tod_turn.context)
            context.system_utterances.append(
                prev_tod_turn.context.next_system_utterance
            )
            context.user_utterances.append(prev_tod_turn.context.current_user_utterance)
            context.prev_tod_turn = prev_tod_turn

        if user_turn:
            utterance = user_turn.utterance
            if self.cfg.delexicalize:
                utterance = self._delexicalize_utterance(user_turn, schemas)
            context.current_user_utterance = utterance
        if system_turn:
            utterance = system_turn.utterance
            if self.cfg.delexicalize:
                utterance = self._delexicalize_utterance(system_turn, schemas)
            context.next_system_utterance = utterance
            if self.cfg.should_add_service_results:
                if len(system_turn.frames) > 1:
                    raise ValueError("More than one frame in system turn")
                for frame in system_turn.frames:
                    context.service_results = frame.service_results
                    # context.service_call = frame.service_call
        return context

    def _prepare_turn(
        self,
        user_turn,
        system_turn,
        prev_tod_turn,
        schemas,
        services,
        woz_user_acts,
        woz_sys_acts,
    ) -> ZsTodTurn:
        turn_schemas = None
        turn_schema_str = None
        if self.cfg.should_add_schema:
            turn_schemas = [schemas[s] for s in services]
            turn_schema_str = "".join([str(s) for s in turn_schemas])
        context = self._prepare_context(
            user_turn, system_turn, prev_tod_turn, schemas, woz_user_acts, woz_sys_acts
        )
        target = self._prepare_target(
            user_turn, system_turn, schemas, woz_user_acts, woz_sys_acts
        )
        return ZsTodTurn(
            context, target, schemas=turn_schemas, schema_str=turn_schema_str
        )

    def _prepare_dialog(
        self,
        dialog: DstcDialog,
        schemas: dict[str, MultiWozSchema],
        turn_csv_row_handler: TurnCsvRowBase,
        act: MultiWozDialogAct,
    ) -> list[ZsTodTurn]:
        tod_turns = []
        tod_turn = None
        if not self._is_dialogue_in_domain(dialog.services):
            return None

        for i, (user_turn, system_turn) in enumerate(utils.grouper(dialog.turns, 2)):
            woz_user_acts = act.turns[str(i * 2)]
            woz_sys_acts = act.turns[str(i * 2 + 1)]
            tod_turn = self._prepare_turn(
                user_turn,
                system_turn,
                tod_turn,
                schemas,
                dialog.services,
                woz_user_acts,
                woz_sys_acts,
            )
            tod_turn.dialog_id = dialog.dialogue_id
            tod_turn.turn_id = i + 1
            tod_turn.active_intent = user_turn.get_active_intent()
            if self.cfg.is_multi_task:
                tod_turns.append(
                    self._prepare_multitask_dialog(tod_turn, turn_csv_row_handler)
                )
            else:
                tod_turns.append(
                    turn_csv_row_handler.to_csv_row(
                        self.cfg.context_type, tod_turn, self.cfg.should_add_schema
                    )
                )
        if not self.cfg.is_multi_task:
            return tod_turns
        out = np.concatenate(tod_turns, axis=0)
        return out

    def _prepare_dialog_file(
        self,
        path: Path,
        schemas: dict[str, MultiWozSchema],
        turn_csv_row_handler: TurnCsvRowBase,
        acts: dict[str, MultiWozDialogAct],
    ):
        data = []
        dialog_json_data = utils.read_json(path)
        for d in dialog_json_data:
            dialog = DstcDialog.from_dict(d)
            prepped_dialog = self._prepare_dialog(
                dialog, schemas, turn_csv_row_handler, acts[dialog.dialogue_id]
            )
            if prepped_dialog is None:
                continue
            data.append(prepped_dialog)
        if not len(data):
            return np.array(data)
        return np.concatenate(data, axis=0)

    def run(self):
        steps = Steps.list()
        schemas = self.get_schemas()
        acts = self.get_acts()
        step_dir = self.cfg.processed_data_root / self.cfg.step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        dialog_paths = utils.get_dialog_file_paths(
            self.cfg.raw_data_root, self.cfg.step_name
        )
        turn_csv_row_handler: TurnCsvRowBase = TurnCsvRowFactory.get_handler(self.cfg)
        out_data = []
        if self.cfg.num_dialogs == "None":
            self.cfg.num_dialogs = len(dialog_paths)
        csv_file_path = utils.get_csv_data_path(
            step=self.cfg.step_name,
            num_dialogs=self.cfg.num_dialogs,
            cfg=self.cfg,
        )
        if csv_file_path.exists() and not self.cfg.overwrite:
            print(
                f"{self.cfg.step_name} csv file already exists and overwrite is false, so skipping"
            )
            return

        if self.cfg.data_prep_multi_process:
            res = list(
                tqdm(
                    Pool().imap(
                        self._prepare_dialog_file,
                        dialog_paths[: self.cfg.num_dialogs],
                        itertools.repeat(schemas),
                        itertools.repeat(turn_csv_row_handler),
                        itertools.repeat(acts),
                    ),
                    total=self.cfg.num_dialogs,
                )
            )
        # start no mp code
        else:
            res = []
            for d in tqdm(dialog_paths[: self.cfg.num_dialogs]):
                output = self._prepare_dialog_file(
                    d, schemas, turn_csv_row_handler, acts
                )
                if res is not None:
                    res.append(output)
        # end no mp code

        out_data = [d for d in res if len(d)]
        headers = turn_csv_row_handler.get_csv_headers(self.cfg.should_add_schema)
        if len(out_data) == 0:
            print(f"No data for {self.cfg.step_name}")
            return
        csv_data = np.concatenate(out_data, axis=0)
        utils.write_csv(headers, csv_data, csv_file_path)


@hydra.main(config_path="../../config/data_prep/", config_name="multi_woz_22_data_prep")
def hydra_start(cfg: DictConfig) -> None:
    tmwdp = TodMultiWoz22DataPrep(MultiWozDataPrepConfig(**cfg))
    tmwdp.run()


if __name__ == "__main__":
    hydra_start()
