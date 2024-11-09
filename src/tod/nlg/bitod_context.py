from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Optional

from my_enums import TurnRowType


from tod.nlg.bitod_api_call import BitodApiCall
from tod.nlg.ke_tod_context import KeTodContext


@dataclass
class BiTodContext(KeTodContext):
    def __init__(self, max_length: int = 10, context_formatter: any = None):
        super().__init__(max_length, context_formatter=context_formatter)

    def get_api_call(self) -> str:
        out = ""
        if not self.api_call:
            return out
        dstc_api_call = BitodApiCall(self.api_call.method, self.api_call.parameters)
        return str(dstc_api_call)

    def get_service_results(self, num_items: int = 1) -> str:
        out = ""
        if not self.service_results:
            return out
        return str(dict(self.service_results))
        results = self.service_results[:num_items]
        results = [dict(r) for r in results]
        return str(results)
