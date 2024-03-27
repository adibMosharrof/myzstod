from dataclasses import dataclass, field
from typing import Optional
from sgd_dstc8_data_model.dstc_dataclasses import DstcRequestedSlot

from my_enums import ZsTodConstants, SpecialTokens
from tod.zs_tod_action import ZsTodAction
from tod.zs_tod_dst import ZsTodDst


@dataclass
class ZsTodTarget:
    actions: list[ZsTodAction]
    user_actions: list[ZsTodAction]
    response: str
    dsts: list[ZsTodDst]
    requested_slots: Optional[list[DstcRequestedSlot]] = None

    # these three methods are used in mh dataclasses, so be careful when renaming them
    def get_dsts(self) -> str:
        return "".join(
            [
                SpecialTokens.begin_dsts,
                "".join(map(str, self.dsts)),
                SpecialTokens.end_dsts,
            ]
        )

    def get_actions(self) -> str:
        return "".join(
            [
                SpecialTokens.begin_action,
                ZsTodConstants.ITEM_SEPARATOR.join(map(str, self.actions)),
                SpecialTokens.end_action,
            ]
        )

    def get_response(self) -> str:
        return "".join(
            [
                SpecialTokens.begin_response,
                self.response,
                SpecialTokens.end_response,
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def get_nlg_target_str(self) -> str:
        out = " ".join(
            [
                # "System Response",
                self.response,
            ]
        )
        return out

    def __str__(self) -> str:
        out = "".join(
            [
                SpecialTokens.begin_target,
                SpecialTokens.begin_dsts,
                "".join(map(str, self.dsts)),
                SpecialTokens.end_dsts,
                # ZsTodConstants.NEW_LINES,
                SpecialTokens.begin_user_action,
                ZsTodConstants.ITEM_SEPARATOR.join(map(str, self.user_actions))
                if self.user_actions
                else "",
                SpecialTokens.end_user_action,
                SpecialTokens.begin_action,
                ZsTodConstants.ITEM_SEPARATOR.join(map(str, self.actions))
                if self.actions
                else "",
                SpecialTokens.end_action,
                SpecialTokens.begin_response,
                self.response,
                SpecialTokens.end_response,
                SpecialTokens.end_target,
                # ZsTodConstants.NEW_LINES,
            ]
        )
        return out
