from itertools import zip_longest
from tod.context_formatter.context_formatter_base import ContextFormatterBase
from tod.zs_tod_context import ZsTodContext


class NlgContextFormatter(ContextFormatterBase):

    def to_str(self, context: ZsTodContext) -> str:
        history = []
        for user, system in zip_longest(
            context.user_utterances, context.system_utterances, fillvalue=""
        ):
            if user:
                history.append(f"User: {user}")
            if system:
                history.append(f"System: {system}")
        history_text = "\n".join(history)
        return "\n".join(
            [
                history_text,
                context._get_last_user_utterance(),
                "End Dialog History",
            ]
        )
