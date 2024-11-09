from itertools import zip_longest
from tod.context_formatter.context_formatter_base import ContextFormatterBase
from tod.zs_tod_context import ZsTodContext
from my_enums import SpecialTokens


class SimpleTodContextFormatter(ContextFormatterBase):

    def to_str(self, context: ZsTodContext) -> str:
        history = []
        for user, system in zip_longest(
            context.user_utterances, context.system_utterances, fillvalue=""
        ):
            if user:
                history.append(f"{SpecialTokens.user.value}{user}")
            if system:
                history.append(f"{SpecialTokens.system.value}{system}")
        last_utterance = context.current_user_utterance

        history.append(f"{SpecialTokens.user.value}{last_utterance}")
        history_text = "\n".join(history)
        return "\n".join(
            [
                SpecialTokens.begin_context.value,
                history_text,
                SpecialTokens.end_context.value,
                context.get_service_results(),
            ]
        )
