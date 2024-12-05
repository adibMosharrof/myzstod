from configs.dataprep_config import DataPrepConfig
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
    DstcTurn,
)
from data_prep.ketod.ketod_nlg_api_call_strategy import KetodNlgApiCallStrategy
from my_enums import ZsTodConstants
from tod.nlg.ke_tod_context import KeTodContext
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.turns.zs_tod_turn import ZsTodTurn
from tod.zs_tod_dst import ZsTodDst
from tod.zs_tod_target import ZsTodTarget
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief
from tod.zs_tod_context import ZsTodContext
from utilities import text_utilities
from utilities.dialog_studio_dataclasses import Log


class ZsKetodDataPrepStrategy(KetodNlgApiCallStrategy):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg, tod_turn_cls=NlgTodTurn, tod_context_cls=KeTodContext)

    def _prepare_dst(self, user_turn: DstcTurn) -> list[ZsTodBelief]:
        dsts = []
        for frame in user_turn.frames:
            if not frame.state:
                continue
            beliefs = []
            active_intent = frame.state.active_intent
            short_service_name = text_utilities.get_nlg_service_name(frame.service)
            requested_slots = [
                "".join(
                    [
                        short_service_name,
                        ZsTodConstants.DOMAIN_SLOT_SEPARATOR,
                        slot,
                    ]
                )
                for slot in frame.state.requested_slots
            ]
            for slot_name, value in frame.state.slot_values.items():
                beliefs.append(
                    ZsTodBelief(
                        short_service_name,
                        # humps.camelize(slot_name),
                        slot_name,
                        value,
                    )
                )
            dsts.append(ZsTodDst(beliefs, active_intent, requested_slots))
        return dsts

    def _get_actions(self, turn: DstcTurn) -> list[ZsTodAction]:
        actions = []
        for frame in turn.frames:
            short_service_name = text_utilities.get_nlg_service_name(frame.service)
            for action in frame.actions:
                actions.append(
                    ZsTodAction(
                        short_service_name,
                        action.act,
                        action.slot,
                        # Note: Dotmap has a built-in values() method, so accessing the values in dict format.
                        ZsTodConstants.ACTION_VALUE_SEPARATOR.join(action["values"]),
                    )
                )
        return actions

    def prepare_target(
        self,
        turn: Log,
        schemas: dict[str, DstcSchema],
    ) -> ZsTodTarget:
        dsts = self._prepare_dst(turn.original_user_side_information)
        actions = self._get_actions(turn.original_system_side_information)
        user_actions = (
            self._get_actions(turn.original_user_side_information)
            if self.cfg.should_add_user_actions
            else None
        )
        response = self._prepare_response(turn.system_response)
        return ZsTodTarget(
            dsts=dsts, actions=actions, user_actions=user_actions, response=response
        )

    def get_turn_schema_str(self, turn_schemas) -> str:
        for schema in turn_schemas:
            for slot in schema.slots:
                slot.possible_values = []

        return "".join([str(s) for s in turn_schemas])
