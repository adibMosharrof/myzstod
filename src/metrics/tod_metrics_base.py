from abc import ABC
import abc
from pathlib import Path
import re
from typing import Optional, Union

import numpy as np

from predictions_logger import PredictionsLoggerBase
from my_enums import ZsTodConstants
import dstc.dstc_utils as dstc_utils
from torchmetrics import Metric


# class TodMetricsBase(ABC):
class TodMetricsBase(Metric):
    """Base class for all TOD metrics."""

    full_state_update = False

    def __init__(
        self,
        score: bool = 0.0,
        is_cached=False,
        prediction_logger: PredictionsLoggerBase = None,
    ):
        super().__init__()
        # super().full_state_update = full_state_update
        self.score = score
        self.is_cached = is_cached
        self.wrong_preds = {}
        self.prediction_logger = prediction_logger

    def _add_wrong_pred(self, key: any):
        if type(key) not in [int, str]:
            key = str(key)
        self.wrong_preds[key] = self.wrong_preds.get(key, 0) + 1

    def _log_prediction(
        self, pred: str = None, ref: str = None, is_correct: any = None
    ):
        self.prediction_logger.log(pred, ref, is_correct)

    def visualize(self, out_dir: Path) -> None:
        if not self.prediction_logger:
            return
        self.prediction_logger.visualize(out_dir)

    def _extract_section_from_text(
        self,
        text: Union[str, list[str]],
        start_token: str,
        end_token: str,
        default_value: any = None,
        multiple_values: bool = False,
        trim_spaces: bool = False,
    ):
        text = dstc_utils.get_text_in_between(
            text, start_token, end_token, default_value, multiple_values=multiple_values
        )
        if not trim_spaces:
            return text
        if isinstance(text, list):
            return [t.strip() for t in text]
        return text.strip()

    def _extract_section_and_split_items_from_text(
        self,
        text: str,
        start_token: str,
        end_token: str,
        separator: str = ZsTodConstants.ITEM_SEPARATOR,
        default_value: any = [],
        multiple_values: bool = False,
        trim_spaces: bool = False,
    ) -> np.ndarray:
        section_txts = self._extract_section_from_text(
            text,
            start_token,
            end_token,
            default_value,
            multiple_values=multiple_values,
            trim_spaces=trim_spaces,
        )
        if not section_txts:
            return default_value
        if type(section_txts) == list:
            out = [st.split(separator) for st in section_txts]
            return np.concatenate(out, axis=0, dtype=str)
        return np.array(section_txts.split(separator), dtype=str)

        # return section_txts.split(separator)

    def update(self, predictions: list[str], references: list[str]) -> None:
        if not len(predictions):
            raise ValueError("You must provide at least one prediction.")
        if not len(references):
            raise ValueError("You must provide at least one reference.")
        if not len(predictions) == len(references):
            raise ValueError(
                f"Predictions {len(predictions)} and references {len(references)} must have the same length"
            )
        self.is_cached = False
        return self._update(predictions, references)

    @abc.abstractmethod
    def _update(self, predictions: list[str], references: list[str]) -> None:
        pass

    def compute(self) -> float:
        if self.is_cached:
            return self.score
        self.score = self._compute()
        self.is_cached = True
        return self.score

    @abc.abstractmethod
    def _compute(self) -> float:
        pass


class MetricCollection:
    """Collects multiple metrics.
    Args:
        metrics: A dictionary of metrics.
    Example Usage:
        metrics = MetricCollection(
            {
                "goal_accuracy": GoalAccuracyMetric(),
                "intent_accuracy": IntentAccuracyMetric(),
                "requested_slots": RequestedSlotsMetric(),
            }
        )
        references = # list of whole target str
        predictions = # list of whole prediction str
        metrics.add_batch(predictions, references)
    """

    def __init__(self, metrics: dict[str, TodMetricsBase] = None):
        if metrics is None:
            raise ValueError("No metrics provided to MetricCollection")
        self.metrics = metrics

    def add_batch(self, predictions: list[str], references: list[str]) -> None:
        for m in self.metrics.values():
            m.update(predictions, references)

    def compute(self) -> float:
        return [m.compute() for m in self.metrics.values()]

    def visualize(self, out_dir: Path) -> None:
        return [m.visualize(out_dir) for m in self.metrics.values()]

    def __str__(self):
        return "\n".join([str(m) for m in self.metrics.values()])
