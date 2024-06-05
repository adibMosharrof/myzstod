from metric_managers.bitod_metric_manager import BitodMetricManager
from metric_managers.ketod_metric_manager import KeTodMetricManager
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from metric_managers.nlg_metric_manager import NlgMetricManager
from my_enums import ContextType


class MetricManagerFactory:

    @classmethod
    def get_metric_manager(self, context_type: str, tokenizer, logger):
        if context_type == ContextType.NLG_API_CALL.value:
            return NlgApiCallMetricManager(logger, tokenizer)
        if context_type == ContextType.KETOD_API_CALL.value:
            return KeTodMetricManager(logger, tokenizer)
        if context_type == ContextType.BITOD.value:
            return BitodMetricManager(logger, tokenizer)
        return NlgMetricManager(logger, tokenizer)
