import abc
from dataclasses import dataclass
from enum import Enum
from fileinput import filename
from pathlib import Path
from tokenize import group
from typing import Optional

import humps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numerize.numerize import numerize
from sklearn.metrics import confusion_matrix

from simple_tod_dataclasses import GoalMetricConfigType, SimpleTodBelief
from dotmap import DotMap

logger_cols = DotMap(
    PREDICTIONS="predictions",
    REFERENCES="references",
    IS_CORRECT="is_correct",
    COUNT="count",
)


@dataclass
class StackedBarChartData:
    df: pd.DataFrame
    df_false: pd.DataFrame
    error_refs: pd.DataFrame
    top_error_refs: pd.DataFrame
    proportions: pd.DataFrame
    counts: pd.DataFrame


class PredictionsLoggerBase(abc.ABC):
    def __init__(self):
        self.refs = []
        self.preds = []
        self.is_correct = []

    @abc.abstractmethod
    def log(self, pred: any = None, ref: any = None, is_correct: any = None):
        raise (NotImplementedError)

    @abc.abstractmethod
    def visualize(self, out_dir: Path):
        raise (NotImplementedError)

    def plot_confusion_matrix(self, labels, x_label, y_label, title, file_name):
        plt.figure(figsize=(10, 10), dpi=200)
        cf_matrix = confusion_matrix(self.refs, self.preds, labels=labels)
        annot_formatter = np.vectorize(lambda x: numerize(int(x), 1), otypes=[np.str])
        annotations = annot_formatter(cf_matrix)
        sns.heatmap(
            cf_matrix,
            annot=annotations,
            fmt="",
            linewidths=1,
            cmap="rocket_r",
            annot_kws={"fontsize": 8 if cf_matrix.shape[0] < 20 else 6},
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        ticks = np.arange(len(labels)) + 0.5
        plt.xticks(
            fontsize=8,
            rotation=90,
            labels=labels,
            ticks=ticks,
        )
        plt.yticks(
            fontsize=8,
            rotation=0,
            labels=labels,
            ticks=ticks,
        )
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    def _get_stacked_bar_chart_data(
        self,
        group_columns=[logger_cols.REFERENCES],
        bar_group_column=logger_cols.REFERENCES,
        top_k=10,
        df=None,
    ) -> StackedBarChartData:
        group_by = np.hstack([group_columns, [logger_cols.IS_CORRECT]]).tolist()
        df_false = (
            df[df[logger_cols.IS_CORRECT] == False]
            .groupby(group_by)[logger_cols.IS_CORRECT]
            .count()
            .reset_index(name=logger_cols.COUNT)
        )
        error_refs = (
            df_false.groupby(group_columns)[logger_cols.COUNT]
            .sum()
            .reset_index(name=logger_cols.COUNT)
            .sort_values(logger_cols.COUNT, ascending=False)
        )

        # stacked hbar plot
        top_error_refs_bar = error_refs[:top_k]
        df_bar = df[df[bar_group_column].isin(top_error_refs_bar[bar_group_column])]
        # sorting values by predictions where is_correct is false
        data_prop = pd.crosstab(
            index=df_bar[bar_group_column],
            columns=df_bar[logger_cols.IS_CORRECT],
            normalize="index",
        ).sort_values(False)

        data_count = pd.crosstab(
            index=df_bar[bar_group_column],
            columns=df_bar[logger_cols.IS_CORRECT],
        ).reindex(data_prop.index)

        return StackedBarChartData(
            df, df_false, error_refs, top_error_refs_bar, data_prop, data_count
        )

    def _plot_stacked_bar_chart(
        self, data: StackedBarChartData, x_label="", y_label="", title="", file_name=""
    ):

        plt.style.use("ggplot")
        sns.set(style="darkgrid")

        plt.figure(figsize=(10, 15), dpi=200)
        data.proportions.plot(
            kind="barh",
            stacked=True,
            figsize=(10, 15),
            fontsize=8,
            color=["r", "g"],
            width=0.8,
        )
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        for n, x in enumerate([*data.counts.index.values]):
            for proportion, count, y_loc in zip(
                data.proportions.loc[x],
                data.counts.loc[x],
                data.proportions.loc[x].cumsum(),
            ):

                plt.text(
                    y=n - 0.035,
                    x=(y_loc - proportion) + (proportion / 2),
                    s=f"   {numerize(count,1)}\n{proportion*100:.1f}%",
                    fontweight="bold",
                    fontsize=8,
                    color="black",
                )
        plt.title(title)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()


"""
TODO: seems a little complex, so on hold
extend generic logger class
return data from visualize method
call super visualize
plot confusion matrix
"""


class IntentsPredictionLogger(PredictionsLoggerBase):
    def __init__(self):
        super().__init__()

    def log(self, pred, ref, is_correct):
        self.preds.append(pred)
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path, top_k_bar=10, top_k_confusion=5):
        df = pd.DataFrame(
            {
                logger_cols.REFERENCES: self.refs,
                logger_cols.IS_CORRECT: self.is_correct,
            }
        )
        if len(self.preds):
            df = pd.concat(
                [df, pd.DataFrame({logger_cols.PREDICTIONS: self.preds})], axis=1
            )
        data = self._get_stacked_bar_chart_data(
            df=df,
            top_k=top_k_bar,
            group_columns=[logger_cols.REFERENCES, logger_cols.PREDICTIONS],
            bar_group_column=logger_cols.REFERENCES,
        )

        self._plot_stacked_bar_chart(
            data,
            "Proportion",
            "Intents",
            "Intents Predictions",
            out_dir / "intent_predictions.png",
        )

        top_error_refs_cm = list(data.proportions.index[:top_k_confusion])
        heatmap_data = data.df_false[
            data.df_false[logger_cols.REFERENCES].isin(top_error_refs_cm)
        ]
        cf_labels = np.unique(
            [
                heatmap_data[logger_cols.REFERENCES],
                heatmap_data[logger_cols.PREDICTIONS],
            ]
        )

        self.plot_confusion_matrix(
            cf_labels,
            humps.camelize(logger_cols.PREDICTIONS),
            humps.camelize(logger_cols.REFERENCES),
            "Intents Confusion Matrix",
            out_dir / "intents_confusion_matrix.png",
        )


class GenericPredictionLogger(PredictionsLoggerBase):
    def __init__(self, columns: list[str], metric_name: str):
        super().__init__()
        self.columns = columns
        self.metric_name = metric_name

    def log(self, pred=None, ref=None, is_correct=None):
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path):
        if not len(self.refs):
            return
        df = pd.concat(
            [
                pd.DataFrame(map(lambda x: x.__dict__, self.refs)),
                pd.DataFrame(self.is_correct, columns=[logger_cols.IS_CORRECT]),
            ],
            axis=1,
        )
        for col in self.columns:
            data = self._get_stacked_bar_chart_data(
                df=df, group_columns=[col], bar_group_column=col
            )
            self._plot_stacked_bar_chart(
                data,
                "Proportion",
                f"{humps.pascalize(self.metric_name)} {humps.pascalize(col)}s",
                f"{humps.pascalize(self.metric_name)}{humps.pascalize(col)} Predictions",
                out_dir / f"{self.metric_name}_{col}_predictions.png",
            )


class TodMetricsEnum(str, Enum):
    BELIEF = "belief"
    ACTION = "action"
    INFORM = "inform"
    REQUESTED_SLOTS = "requested_slots"
    SUCCESS = "success"


class PredictionLoggerFactory:
    @staticmethod
    def create(metric: TodMetricsEnum) -> PredictionsLoggerBase:
        if metric in [TodMetricsEnum.INFORM, TodMetricsEnum.REQUESTED_SLOTS]:
            columns = ["domain", "slot_name"]
        elif metric in [TodMetricsEnum.ACTION, TodMetricsEnum.SUCCESS]:
            columns = ["domain", "action_type", "slot_name", "values"]
        elif metric == TodMetricsEnum.BELIEF:
            columns = ["domain", "slot_name", "value"]

        return GenericPredictionLogger(metric_name=metric.value, columns=columns)
