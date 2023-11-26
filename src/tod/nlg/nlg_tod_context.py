from collections import deque
from dataclasses import dataclass
from itertools import zip_longest
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NlgTodContext:
    user_utterances: deque[str] = field(default_factory=deque)
    system_utterances: deque[str] = field(default_factory=deque)
    next_system_utterance: str = None
    current_user_utterance: str = None
    should_add_sys_actions: bool = None
    prev_tod_turn: Optional[any] = None
    service_results: Optional[list[dict[str, str]]] = None
    service_call: Optional[dict[str, str]] = None

    def __init__(self, max_length: int = 10):
        self.user_utterances = deque(maxlen=max_length)
        self.system_utterances = deque(maxlen=max_length)

    def _get_last_user_utterance(self) -> str:
        return "".join(["Last User Utterance:", self.current_user_utterance])

    def _get_service_call(self) -> str:
        out = ""
        if not self.service_call:
            return out
        self.service_call.__class__.__qualname__ = "ServiceCall"
        out += str(self.service_call)
        return out

    def _get_service_results(self) -> str:
        out = ""
        if not self.service_results:
            return out
        for service_result in self.service_results[:1]:
            s_res = {"search_results": service_result}
            out += str(s_res)
        return "\n".join(["\nSearch Results:", out, "End Search Results"])

    def __str__(self):
        history = []
        for user, system in zip_longest(
            self.user_utterances, self.system_utterances, fillvalue=""
        ):
            if user:
                history.append(f"User: {user}")
            if system:
                history.append(f"System: {system}")
        history_text = "\n".join(history)
        return "\n".join(
            [
                history_text,
                self._get_last_user_utterance(),
                "End Dialog History",
                self._get_service_results(),
                self._get_service_call(),
            ]
        )
