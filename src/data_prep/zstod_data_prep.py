from configs.dataprep_config import DataPrepConfig
from sgd_dstc8_data_model.dstc_dataclasses import (
    DstcSchema,
    DstcTurn,
)
from data_prep.data_prep_strategy import DataPrepStrategy
from my_enums import Steps, ZsTodConstants
from tod.turns.zs_tod_turn import ZsTodTurn
from tod.zs_tod_dst import ZsTodDst
from tod.zs_tod_target import ZsTodTarget
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief
from tod.zs_tod_context import ZsTodContext


class ZsTodDataPrep(DataPrepStrategy):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg, tod_turn_cls=ZsTodTurn, tod_context_cls=ZsTodContext)

    def _prepare_dst(self, user_turn: DstcTurn) -> list[ZsTodBelief]:
        dsts = []
        for frame in user_turn.frames:
            if not frame.state:
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

    def _get_actions(self, turn: DstcTurn) -> list[ZsTodAction]:
        actions = []
        for frame in turn.frames:
            for action in frame.actions:
                actions.append(
                    ZsTodAction(
                        frame.short_service,
                        action.act,
                        action.slot,
                        ZsTodConstants.ACTION_VALUE_SEPARATOR.join(action.values),
                    )
                )
        return actions

    def prepare_target(
        self,
        user_turn: DstcTurn,
        system_turn: DstcTurn,
        schemas: dict[str, DstcSchema],
    ) -> ZsTodTarget:
        dsts = self._prepare_dst(user_turn)
        actions = self._get_actions(system_turn)
        user_actions = (
            self._get_actions(user_turn) if self.cfg.should_add_user_actions else None
        )
        response = self._prepare_response(system_turn.utterance)

        return ZsTodTarget(
            dsts=dsts, actions=actions, user_actions=user_actions, response=response
        )

    def get_turn_schema_str(self, turn_schemas) -> str:
        for schema in turn_schemas:
            for slot in schema.slots:
                slot.possible_values = []

        return "".join([str(s) for s in turn_schemas])
