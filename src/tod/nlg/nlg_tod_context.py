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
    api_call: Optional[dict[str, str]] = None

    def __init__(self, max_length: int = 10):
        self.user_utterances = deque(maxlen=max_length)
        self.system_utterances = deque(maxlen=max_length)

    def _get_last_user_utterance(self) -> str:
        if self.current_user_utterance:
            return "".join(["Last User Utterance:", self.current_user_utterance])
        return ""

    def get_api_call(self, schemas, turn_domains) -> str:
        out = ""
        if not self.api_call:
            return out
        self.api_call.__class__.__qualname__ = "ApiCall"
        out += str(self.api_call)
        return out

    def get_service_results(self, num_items: int = 1) -> str:
        out = ""
        if not self.service_results:
            return out
        results = self.service_results[:num_items]
        return str(results)
        for service_result in self.service_results[:1]:
            s_res = {"search_results": service_result}
            out += str(s_res)
        return "\n".join([out])

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
                # self.get_service_results(),
                # self.get_api_call(),
            ]
        )
