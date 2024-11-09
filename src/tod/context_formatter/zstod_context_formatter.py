from itertools import zip_longest
from tod.context_formatter.context_formatter_base import ContextFormatterBase
from tod.zs_tod_context import ZsTodContext
from my_enums import SpecialTokens


class ZsTodContextFormatter(ContextFormatterBase):

    def to_str(self, context: ZsTodContext) -> str:
        return "".join(
            [
                SpecialTokens.begin_context,
                (
                    context.prev_tod_turn.target.get_dsts()
                    if context.prev_tod_turn
                    else ""
                ),
                context.get_service_results(),
                context._get_sys_actions(),
                context._get_last_user_utterance(),
                SpecialTokens.end_context,
            ]
        )
