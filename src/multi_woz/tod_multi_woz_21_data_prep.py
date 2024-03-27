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
from tod.turns.turn_csv_row_base import TurnCsvRowBase
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


class TodMultiWoz21DataPrep:
    def __init__(self, cfg: MultiWozDataPrepConfig):
        self.cfg = cfg

    def _get_dialog_services(self, dialog: dict[str, any]) -> list[str]:
        dialog_services = []
        for key, val in dialog.items():
            if key == "log":
                continue
            for domain, item in val.items():
                if domain in ["topic", "message"]:
                    continue
                if bool(item):
                    dialog_services.append(domain)
        return dialog_services

    def _is_dialogue_in_domain(self, dialog: dict[str:any], dialog_services) -> bool:
        return all(ds in self.cfg.domains for ds in dialog_services)

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
            if isinstance(turn, str):
                continue
            for key, values in turn.items():
                if key == "span_info":
                    continue
                domain, action_type = key.split("-")
                actions = []
                for slot_name, slot_value in values:
                    # for act_name, act_values in values.items():
                    actions.append(
                        MultiWozAct(
                            action_type=action_type,
                            slot_name=slot_name,
                            value=slot_value,
                        )
                    )
            out.turns[turn_id] = actions
        return out

    def get_acts(self) -> dict[str, MultiWozDialogAct]:
        file_path = self.cfg.raw_data_root / "system_acts.json"
        acts_json = utils.read_json(file_path)
        woz_acts = {}
        for dialog_id, acts in acts_json.items():
            row = self._read_act(dialog_id, acts)
            woz_acts[dialog_id] = row
        return woz_acts

    def _prepare_dst(self, turn: DstcTurn) -> list[ZsTodBelief]:
        dsts = []
        for key, val in turn["dialog_act"].items():
            if "Inform" not in key:
                continue
            domain, action_type = key.split("-")
            beliefs = []
            for items in val:
                slot_name, value = items
                beliefs.append(
                    ZsTodBelief(
                        domain,
                        slot_name,
                        [value],
                    )
                )

            dsts.append(ZsTodDst(beliefs, None, None))
        return dsts

    def _get_actions(self, sys_turn, woz_acts: MultiWozAct) -> list[ZsTodAction]:
        actions = []

        for key, value in sys_turn["dialog_act"].items():
            if not any([x in key for x in ["Inform", "Request"]]):
                continue
            domain, action_type = key.split("-")
            for val in value:
                slot_name, slot_value = val
                actions.append(
                    ZsTodAction(
                        domain,
                        action_type,
                        slot_name,
                        slot_value,
                    )
                )
        return actions

    def _prepare_response(
        self, system_turn: DstcTurn, schemas: dict[str, MultiWozSchema]
    ) -> str:
        if not system_turn:
            return None
        if not self.cfg.delexicalize:
            return system_turn["text"]
        return self._delexicalize_utterance(system_turn, schemas)

    def _prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, MultiWozSchema],
        woz_user_acts: list[MultiWozAct],
        woz_sys_acts: list[MultiWozAct],
    ) -> ZsTodTarget:
        dsts = self._prepare_dst(user_turn) if "dialog_act" in user_turn else []
        actions = (
            self._get_actions(system_turn, woz_sys_acts)
            if "dialog_act" in system_turn
            else []
        )
        user_actions = (
            self._get_actions(user_turn, woz_user_acts)
            if "dialog_act" in user_turn
            else []
        )
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
            utterance = user_turn["text"]
            if self.cfg.delexicalize:
                utterance = self._delexicalize_utterance(user_turn, schemas)
            context.current_user_utterance = utterance
        if system_turn:
            utterance = system_turn["text"]
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
        dialog_id: str,
    ) -> list[ZsTodTurn]:
        tod_turns = []
        tod_turn = None
        dialog_services = self._get_dialog_services(dialog)
        if not self._is_dialogue_in_domain(dialog, dialog_services):
            return None

        for i, (user_turn, system_turn) in enumerate(utils.grouper(dialog["log"], 2)):
            woz_user_acts = act.turns[str(i * 2)] if act else None
            woz_sys_acts = act.turns[str(i * 2 + 1)] if act else None
            tod_turn = self._prepare_turn(
                user_turn,
                system_turn,
                tod_turn,
                schemas,
                dialog_services,
                woz_user_acts,
                woz_sys_acts,
            )
            tod_turn.dialog_id = dialog_id
            tod_turn.turn_id = i + 1
            # tod_turn.active_intent = user_turn.get_active_intent()
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
        dialog_id: Path,
        schemas: dict[str, MultiWozSchema],
        turn_csv_row_handler: TurnCsvRowBase,
        data_json: any,
        acts: dict[str, MultiWozDialogAct],
    ):
        data = []
        dialog = data_json[dialog_id]
        prepped_dialog = self._prepare_dialog(
            dialog, schemas, turn_csv_row_handler, acts.get(dialog_id), dialog_id
        )
        if prepped_dialog is None:
            return
        data.append(prepped_dialog)
        if not len(data):
            return np.array(data)
        return np.concatenate(data, axis=0)

    def get_dialog_file_names(
        self, data_root: Path, step_name: str, data_json
    ) -> list[str]:
        file_path = data_root / "valListFile.txt"
        dev_files = utils.read_lines_in_file(file_path)
        file_path = data_root / "testListFile.txt"
        test_files = utils.read_lines_in_file(file_path)
        if step_name == Steps.DEV.value:
            return dev_files
        if step_name == Steps.TEST.value:
            return test_files
        all_dialog_names = set(data_json.keys())
        train_dialog_names = all_dialog_names - set(dev_files) - set(test_files)
        return list(train_dialog_names)

    def run(self):
        steps = Steps.list()
        # schemas = self.get_schemas()
        schemas = None
        data_path = self.cfg.raw_data_root / "data.json"
        data_json = utils.read_json(data_path)
        acts = self.get_acts()
        step_dir = self.cfg.processed_data_root / self.cfg.step_name
        step_dir.mkdir(parents=True, exist_ok=True)
        dialog_ids = self.get_dialog_file_names(
            self.cfg.raw_data_root, self.cfg.step_name, data_json
        )
        turn_csv_row_handler: TurnCsvRowBase = TurnCsvRowFactory.get_handler(self.cfg)
        out_data = []
        if self.cfg.num_dialogs == "None":
            self.cfg.num_dialogs = len(dialog_ids)
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
                        dialog_ids[: self.cfg.num_dialogs],
                        itertools.repeat(schemas),
                        itertools.repeat(turn_csv_row_handler),
                        itertools.repeat(data_json),
                        itertools.repeat(acts),
                    ),
                    total=self.cfg.num_dialogs,
                )
            )
        # start no mp code
        else:
            res = []
            for d in tqdm(dialog_ids[: self.cfg.num_dialogs]):
                output = self._prepare_dialog_file(
                    d, schemas, turn_csv_row_handler, data_json, acts
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


@hydra.main(config_path="../../config/data_prep/", config_name="multi_woz_21_data_prep")
def hydra_start(cfg: DictConfig) -> None:
    tmwdp = TodMultiWoz21DataPrep(MultiWozDataPrepConfig(**cfg))
    tmwdp.run()


if __name__ == "__main__":
    hydra_start()
