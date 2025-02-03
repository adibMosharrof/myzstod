from abc import ABC, abstractmethod
import copy
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcDialog,
    DstcFrame,
    DstcSchema,
    DstcTurn,
    get_schemas,
)
from tod.context_formatter.context_formatter_factory import ContextFormatterFactory
from tod.nlg.nlg_tod_context import NlgTodContext
from tod.nlg.nlg_tod_target import NlgTodTarget
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.turn_csv_row_base import TurnCsvRowBase
import data_prep.data_prep_utils as data_prep_utils
from typing import Optional, Union
from tod.turns.zs_tod_turn import ZsTodTurn
from tod.zs_tod_context import ZsTodContext
from tod.zs_tod_target import ZsTodTarget
import utils
from utilities.text_utilities import get_nlg_service_name
from my_enums import ContextType, DstcSystemActions, SpecialTokens
from utilities.context_manager import ContextManager


class DataPrepStrategy(ABC):
    """
    Abstract base class for data preparation strategies.
    """

    def __init__(
        self,
        cfg,
        tod_turn_cls=ZsTodTurn,
        tod_context_cls=ZsTodContext,
        data_augmentations=None,
    ):
        self.cfg = cfg
        self.tod_turn_cls = tod_turn_cls
        self.tod_context_cls = tod_context_cls
        self.data_augmentations = data_augmentations or {}
        self.context_formatter = ContextFormatterFactory.create_context_formatter(
            cfg.context_type
        )

    @abstractmethod
    def prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, DstcSchema],
    ) -> Union[NlgTodTarget, ZsTodTarget]:
        raise NotImplementedError()

    def get_turn_schema_str(self, turn_schemas) -> str:
        return "\n".join([s.get_nlg_repr() for s in turn_schemas])

    def _prepare_response(self, utterance: str) -> str:
        # if ContextManager.is_decoder_type(self.cfg.context_type):
        #     utterance += SpecialTokens.eos_token.value
        # utterance += SpecialTokens.eos_token.value
        return utterance

    def prepare_dialog(
        self,
        dstc_dialog: DstcDialog,
        schemas: dict[str, DstcSchema],
        turn_csv_row_handler: TurnCsvRowBase,
    ) -> Optional[list[NlgTodTurn]]:
        tod_turns = []
        tod_turn = None
        if not data_prep_utils.is_dialogue_in_domain(
            dstc_dialog.services, self.cfg.domains
        ):
            return None

        for i, (user_turn, system_turn) in enumerate(
            utils.grouper(dstc_dialog.turns, 2)
        ):
            tod_turn = self.prepare_turn(
                user_turn, system_turn, tod_turn, schemas, dstc_dialog.services
            )
            tod_turn.dialog_id = dstc_dialog.dialogue_id
            tod_turn.turn_id = i + 1
            tod_turns.append(
                turn_csv_row_handler.to_csv_row(
                    self.cfg.context_type,
                    tod_turn,
                    self.cfg.should_add_schema,
                )
            )
        return tod_turns

    def prepare_turn(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        prev_tod_turn: Union[NlgTodTurn, ZsTodTurn],
        schemas: dict[str, DstcSchema],
        services: list[str],
    ) -> Union[NlgTodTurn, ZsTodTurn]:
        turn_schemas = [schemas[s] for s in services]
        turn_schema_str = self.get_turn_schema_str(turn_schemas)
        context = self.prepare_context(user_turn, system_turn, prev_tod_turn, schemas)
        target = self.prepare_target(user_turn, system_turn, schemas)
        domains = [get_nlg_service_name(s) for s in services]
        return self.tod_turn_cls(
            context=context,
            target=target,
            schemas=turn_schemas,
            schema_str=turn_schema_str,
            domains=domains,
            domains_original=services,
            dataset_name=self.cfg.dataset_name,
            current_user_utterance=context.current_user_utterance,
        )

    def prepare_context(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        prev_tod_turn: NlgTodTurn,
        schemas: dict[str, DstcSchema],
    ) -> Union[NlgTodContext, ZsTodContext]:
        """
        A context contains a list of user and system turns. The data format expects system turn first, and then user turn.

        In the first turn, system turn is null and there is only a user turn and the system turn is placed in
        the next system utterance of the current context.

        If we have a previous turn, we make a deep copy of it. Check context length by number of turns.
        The system utterance for this turn is the next system utterance of the previous context.
        """
        if not prev_tod_turn:
            context = self.tod_context_cls(
                max_length=self.cfg.num_turns, context_formatter=self.context_formatter
            )
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
            context.current_user_utterance = utterance
        if system_turn:
            utterance = system_turn.utterance
            context.next_system_utterance = utterance
            if self.cfg.should_add_service_results:
                if len(system_turn.frames) > 1:
                    raise ValueError("More than one frame in system turn")
                for frame in system_turn.frames:
                    context.service_results = frame.service_results
                    context.api_call = frame.service_call
                # if prev_tod_turn and not context.service_results:
                #     context.service_results = prev_tod_turn.context.service_results
        return context

    def add_tod_turn(
        self,
        turn_csv_row_handler: TurnCsvRowBase,
        tod_turns: list[NlgTodTurn],
        tod_turn: NlgTodTurn,
        dialog_id: int,
        turn_id: int,
    ):
        tod_turn.dialog_id = dialog_id
        tod_turn.turn_id = turn_id
        tod_turns.append(
            turn_csv_row_handler.to_csv_row(
                self.cfg.context_type,
                tod_turn,
                self.cfg.should_add_schema,
                self.cfg.step_name,
            )
        )

    def has_request_action(self, user_turn: DstcTurn) -> bool:
        """
        Check if the user turn has a REQUEST action.
        """
        for frame in user_turn.frames:
            for action in frame.actions:
                if action.act == DstcSystemActions.REQUEST.value:
                    return True
        return False

    def system_has_request_action(self, system_turn: DstcTurn) -> bool:
        """
        Check if the system turn has a REQUEST action.
        """
        for frame in system_turn.frames:
            for action in frame.actions:
                if action.act == DstcSystemActions.REQUEST.value:
                    return True
        return False
