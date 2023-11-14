from collections import deque
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Optional

from my_enums import DstcSystemActions, SpecialTokens
import utils


@dataclass
class ZsTodContext:
    user_utterances: deque[str] = field(default_factory=deque)
    system_utterances: deque[str] = field(default_factory=deque)
    next_system_utterance: str = None
    current_user_utterance: str = None
    should_add_sys_actions: bool = None
    prev_tod_turn: Optional[any] = None
    service_results: Optional[list[dict[str, str]]] = None

    def __init__(self, max_length: int = 10):
        self.user_utterances = deque(maxlen=max_length)
        self.system_utterances = deque(maxlen=max_length)

    def __repr__(self) -> str:
        return self.__str__()

    def get_short_repr(self) -> str:
        return "".join(
            [
                SpecialTokens.begin_context,
                self.prev_tod_turn.target.get_dsts() if self.prev_tod_turn else "",
                self._get_service_results(),
                self._get_sys_actions(),
                self._get_last_user_utterance(),
                SpecialTokens.end_context,
            ]
        )

    def get_nlg_repr(self):
        history = []
        # out = ""
        for user, system in zip_longest(
            self.user_utterances, self.system_utterances, fillvalue=""
        ):
            if user:
                # out += f"User: {user}\n"
                history.append(f"User: {user}")
            if system:
                # out += f"System: {system}\n"
                history.append(f"System: {system}")
        history_text = "\n".join(history)
        return "\n".join(
            [
                history_text,
                self._get_last_user_utterance(should_add_special_token=False),
                "End Dialog History",
                self._get_service_results(should_add_special_tokens=False),
            ]
        )
        out += "\n".join(
            [
                self._get_last_user_utterance(should_add_special_token=False),
                "End Dialog History",
                self._get_service_results(should_add_special_tokens=False),
            ]
        )
        return out

    def _get_service_results(self, should_add_special_tokens: bool = True) -> str:
        out = ""
        if not self.service_results:
            return out
        if not should_add_special_tokens:
            out += "\nSearch Results:\n"
        for service_result in self.service_results[:1]:
            if should_add_special_tokens:
                out += "".join(
                    [
                        SpecialTokens.begin_service_results,
                        " ".join([" ".join([k, v]) for k, v in service_result.items()]),
                        SpecialTokens.end_service_results,
                    ]
                )
            else:
                out += "\n".join(
                    [
                        ":".join([utils.remove_underscore(k), v])
                        for k, v in service_result.items()
                    ]
                )
        if not should_add_special_tokens:
            out += "\nEnd Search Results\n"
        return out

    def _get_sys_actions(self) -> str:
        if not self.should_add_sys_actions:
            return ""
        return "".join([SpecialTokens.sys_actions, " ".join(DstcSystemActions.list())])

    def _get_last_user_utterance(self, should_add_special_token=True) -> str:
        if not should_add_special_token:
            return "".join(["Last User Utterance:", self.current_user_utterance])
        return "".join(
            [
                SpecialTokens.begin_last_user_utterance,
                self.current_user_utterance,
                SpecialTokens.end_last_user_utterance,
            ]
        )

    def __str__(self) -> str:
        out = SpecialTokens.begin_context
        for user, system in zip_longest(
            self.user_utterances, self.system_utterances, fillvalue=""
        ):
            if user:
                out += SpecialTokens.user + user
            if system:
                out += SpecialTokens.system + system

        out += self._get_service_results()

        out += self._get_sys_actions()
        out += self._get_last_user_utterance()
        out += SpecialTokens.end_context
        return out
