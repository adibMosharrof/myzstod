import copy
import itertools
from pathlib import Path
import sys
import os
import numpy as np

from tqdm import tqdm

sys.path.insert(0, os.path.abspath("./src"))

from configs.dialog_studio_data_prep_config import DialogStudioDataPrepConfig
from tod.zs_target import ZsTodTarget
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief
from tod.zs_tod_dst import ZsTodDst

from tod.turns.zs_tod_turn import ZsTodTurn
from tod.turns.turn_csv_row import TurnCsvRowBase
from tod.turns.turn_csv_row_factory import TurnCsvRowFactory
from tod.zs_tod_context import ZsTodContext

from torch.utils.data import Subset
import hydra
from omegaconf import DictConfig
from my_enums import Steps, ZsTodConstants
import utils

from pathos.multiprocessing import ProcessingPool as Pool
from sgd_dstc8_data_model.dstc_dataclasses import DstcDialog, DstcTurn
from datasets import load_dataset


class TaskMaster2DataPrep:
    def __init__(self, cfg: DialogStudioDataPrepConfig):
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

    # def get_schemas(self) -> dict[str, MultiWozSchema]:
    #     file_path = self.cfg.raw_data_root / "schema.json"
    #     schema_json = utils.read_json(file_path)
    #     out = {}
    #     for s in schema_json:
    #         schema = MultiWozSchema.from_dict(s)
    #         out[schema.service_name] = schema
    #     return out

    def _prepare_dst(self, turn: any) -> list[ZsTodBelief]:
        turn_dsts_text = turn["dst"][1:-1]
        dsts = []
        beliefs = []

        for dst in turn_dsts_text.split(","):
            if not dst:
                continue
            splits = dst.split(" ")
            domain = splits[0]
            slot_name = splits[1]
            value = " ".join(splits[2:])
            beliefs.append(
                ZsTodBelief(
                    domain,
                    slot_name,
                    [value],
                )
            )

        dsts.append(ZsTodDst(beliefs, None, None))
        return dsts

    def _prepare_response(self, system_turn: DstcTurn, schemas: dict[str, any]) -> str:
        if not system_turn:
            return None
        if not self.cfg.delexicalize:
            return system_turn["text"]
        return self._delexicalize_utterance(system_turn, schemas)

    def _prepare_target(
        self,
        turn: DstcTurn,
        schemas: dict[str, any],
    ) -> ZsTodTarget:
        dsts = self._prepare_dst(turn)
        response = turn["system response"]
        return ZsTodTarget(
            actions=None, user_actions=None, dsts=dsts, response=response
        )

    def _prepare_context(
        self,
        turn,
        prev_tod_turn: ZsTodTurn,
        schemas: dict[str, any],
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

        context.current_user_utterance = turn["user utterance"]
        context.next_system_utterance = turn["system response"]

        return context

    def _prepare_turn(
        self,
        turn,
        prev_tod_turn,
        schemas=None,
        services=None,
    ) -> ZsTodTurn:
        turn_schemas = None
        turn_schema_str = None
        if self.cfg.should_add_schema:
            turn_schemas = [schemas[s] for s in services]
            turn_schema_str = "".join([str(s) for s in turn_schemas])
        context = self._prepare_context(turn, prev_tod_turn, schemas)
        target = self._prepare_target(turn, schemas)
        return ZsTodTurn(
            context, target, schemas=turn_schemas, schema_str=turn_schema_str
        )

    def _prepare_dialog(
        self,
        dialog: DstcDialog,
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> list[ZsTodTurn]:
        tod_turns = []
        tod_turn = None
        dialog_id = dialog["dialog index"]
        for turn in dialog["log"]:
            tod_turn = self._prepare_turn(
                turn,
                tod_turn,
            )
            tod_turn.dialog_id = dialog_id
            tod_turn.turn_id = turn["turn id"]
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

    def run(self):
        ds = load_dataset("Salesforce/dialogstudio", self.cfg.dataset_name)
        data = ds["train"]

        turn_csv_row_handler: TurnCsvRowBase = TurnCsvRowFactory.get_handler(self.cfg)
        out_data = []
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

        # subset_data = Subset(data, range(30))
        subset_data = Subset(data, range(len(data)))
        if self.cfg.data_prep_multi_process:
            res = list(
                tqdm(
                    Pool().imap(
                        self._prepare_dialog,
                        subset_data,
                        itertools.repeat(turn_csv_row_handler),
                    ),
                    total=len(subset_data),
                )
            )
        # start no mp code
        else:
            res = []
            # for d in tqdm(data[:30]):
            for d in tqdm(subset_data):
                output = self._prepare_dialog(d, turn_csv_row_handler)
                if res is not None:
                    res.append(output)
        # end no mp code

        headers = turn_csv_row_handler.get_csv_headers(self.cfg.should_add_schema)
        if len(res) == 0:
            print(f"No data for {self.cfg.step_name}")
            return

        csv_data = np.concatenate(res, axis=0)
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)
        utils.write_csv(headers, csv_data, csv_file_path)


@hydra.main(
    config_path="../../config/data_prep/", config_name="task_master_2_data_prep"
)
def hydra_start(cfg: DictConfig) -> None:
    tmwdp = TaskMaster2DataPrep(DialogStudioDataPrepConfig(**cfg))
    tmwdp.run()


if __name__ == "__main__":
    hydra_start()
