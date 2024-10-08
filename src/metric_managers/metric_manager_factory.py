from metric_managers.bitod_metric_manager import BitodMetricManager
from metric_managers.ketod_metric_manager import KeTodMetricManager
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from metric_managers.nlg_metric_manager import NlgMetricManager
from my_enums import ContextType
from utilities.context_manager import ContextManager


class MetricManagerFactory:

    @classmethod
    def get_metric_manager(self, context_type: str, tokenizer, logger):
        if any(
            [
                ContextManager.is_sgd_nlg_api(context_type),
                ContextManager.is_sgd_pseudo_labels(context_type),
            ]
        ):
            return NlgApiCallMetricManager(logger, tokenizer)
        if ContextManager.is_ketod(context_type):
            return KeTodMetricManager(logger, tokenizer)
        if ContextManager.is_bitod(context_type):
            return BitodMetricManager(logger, tokenizer)
        return NlgMetricManager(logger, tokenizer)
