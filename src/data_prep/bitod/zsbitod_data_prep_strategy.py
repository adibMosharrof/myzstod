from configs.dataprep_config import DataPrepConfig
from data_prep.bitod.bitod_strategy import BitodStrategy
from my_enums import ZsTodConstants
from tod.nlg.bitod_context import BiTodContext
from tod.nlg.nlg_tod_turn import NlgTodTurn
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_belief import ZsTodBelief
from tod.zs_tod_dst import ZsTodDst
from tod.zs_tod_target import ZsTodTarget
from utilities.dialog_studio_dataclasses import Log


class ZsBitodDataPrepStrategy(BitodStrategy):
    def __init__(self, cfg: DataPrepConfig):
        super().__init__(cfg, tod_turn_cls=NlgTodTurn, tod_context_cls=BiTodContext)

    def _prepare_dst(self, user_turn: dict):
        dsts = []
        for domain in user_turn.state:
            beliefs = []
            active_intent = self._remove_lang_info(user_turn.active_intent)
            for slot_name in user_turn.state[domain]:
                short_service_name = self._get_domain_name(domain)
                value = self._process_value(user_turn.state[domain][slot_name].value)
                beliefs.append(ZsTodBelief(short_service_name, slot_name, value))
            dsts.append(ZsTodDst(beliefs, active_intent, []))
        return dsts

    def _get_actions(self, turn: dict, user_turn: dict):
        domain = self._get_domain_name(user_turn.active_intent)
        actions = []
        for action in turn.Actions:
            actions.append(
                ZsTodAction(
                    domain,
                    action.act,
                    action.slot,
                    ZsTodConstants.ACTION_VALUE_SEPARATOR.join(
                        map(self._process_value, action.value)
                    ),
                )
            )
        return actions

    def prepare_target(self, turn, schemas):
        dsts = self._prepare_dst(turn.original_user_side_information)
        actions = self._get_actions(
            turn.original_system_side_information, turn.original_user_side_information
        )
        user_actions = []
        if self.cfg.should_add_user_actions:
            user_actions = self._get_actions(
                turn.original_user_side_information, turn.original_user_side_information
            )
        response = self._prepare_response(turn.system_response)
        return ZsTodTarget(
            dsts=dsts, actions=actions, response=response, user_actions=user_actions
        )

    def _prepare_response(self, utterance):
        return utterance

    def _get_domain_name(self, name: str):
        return name.split("_")[0]

    def _remove_lang_info(self, name: str):
        if type(name) != str:
            return name
        return name.replace("_en_US", "")

    def _process_value(self, value):
        if type(value) == int:
            return str(value)
        if type(value) == list:
            value = value[0]
        without_lang = self._remove_lang_info(value)
        return without_lang
