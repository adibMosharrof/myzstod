import abc
from dataclasses import dataclass
from enum import Enum
from fileinput import filename
from pathlib import Path
from typing import Optional

import humps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numerize.numerize import numerize
from sklearn.metrics import confusion_matrix

from simple_tod_dataclasses import SimpleTodBelief


class LoggerColsEnum:
    PREDICTIONS = "predictions"
    REFERENCES = "references"
    IS_CORRECT = "is_correct"
    COUNT = "count"


@dataclass
class StackedBarChartData:
    df: pd.DataFrame
    df_false: pd.DataFrame
    error_refs: pd.DataFrame
    top_error_refs: pd.DataFrame
    proportions: pd.DataFrame
    counts: pd.DataFrame


class PredictionsLoggerBase(abc.ABC):
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
            annot_kws={"fontsize": 8},
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

    def _get_false_predictions(self, df):
        if LoggerColsEnum.PREDICTIONS in df.columns:
            return (
                df[df[LoggerColsEnum.IS_CORRECT] == False]
                .groupby(
                    [
                        LoggerColsEnum.PREDICTIONS,
                        LoggerColsEnum.REFERENCES,
                        LoggerColsEnum.IS_CORRECT,
                    ]
                )[LoggerColsEnum.IS_CORRECT]
                .count()
                .reset_index(name=LoggerColsEnum.COUNT)
            )
        return (
            df[df[LoggerColsEnum.IS_CORRECT] == False]
            .groupby(
                [
                    LoggerColsEnum.REFERENCES,
                    LoggerColsEnum.IS_CORRECT,
                ]
            )[LoggerColsEnum.IS_CORRECT]
            .count()
            .reset_index(name=LoggerColsEnum.COUNT)
        )

    def _get_stacked_bar_chart_data(self, top_k=10) -> StackedBarChartData:
        df = pd.DataFrame(
            {
                LoggerColsEnum.REFERENCES: self.refs,
                LoggerColsEnum.IS_CORRECT: self.is_correct,
            }
        )
        if len(self.preds):
            df = pd.concat(
                [df, pd.DataFrame({LoggerColsEnum.PREDICTIONS: self.preds})], axis=1
            )

        df_false = self._get_false_predictions(df)
        error_refs = (
            df_false.groupby([LoggerColsEnum.REFERENCES])[LoggerColsEnum.COUNT]
            .sum()
            .reset_index(name=LoggerColsEnum.COUNT)
            .sort_values(LoggerColsEnum.COUNT, ascending=False)
        )

        # stacked hbar plot
        top_error_refs_bar = error_refs[:top_k]
        df_bar = df[
            df[LoggerColsEnum.REFERENCES].isin(
                top_error_refs_bar[LoggerColsEnum.REFERENCES]
            )
        ]
        # sorting values by predictions where is_correct is false
        data_prop = pd.crosstab(
            index=df_bar[LoggerColsEnum.REFERENCES],
            columns=df_bar[LoggerColsEnum.IS_CORRECT],
            normalize="index",
        ).sort_values(False)

        data_count = pd.crosstab(
            index=df_bar[LoggerColsEnum.REFERENCES],
            columns=df_bar[LoggerColsEnum.IS_CORRECT],
        ).reindex(data_prop.index)

        return StackedBarChartData(
            df, df_false, error_refs, top_error_refs_bar, data_prop, data_count
        )

    def _plot_stacked_bar_chart(
        self, data: StackedBarChartData, x_label, y_label, title, file_name
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
        plt.tight_layout()
        plt.title(title)
        plt.savefig(file_name)
        plt.close()


class IntentsPredictionLogger(PredictionsLoggerBase):
    def __init__(self):
        self.preds = []
        self.refs = []
        self.is_correct = []

    def log(self, pred, ref, is_correct):
        self.preds.append(pred)
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path, top_k_bar=10, top_k_confusion=5):
        data = self._get_stacked_bar_chart_data()

        self._plot_stacked_bar_chart(
            data,
            "Proportion",
            "Intents",
            "Intents Accuracy",
            out_dir / "intents_prediction_distribution.png",
        )

        top_error_refs_cm = list(data.proportions.index[:top_k_confusion])
        heatmap_data = data.df_false[
            data.df_false[LoggerColsEnum.REFERENCES].isin(top_error_refs_cm)
        ]
        cf_labels = np.unique(
            [
                heatmap_data[LoggerColsEnum.REFERENCES],
                heatmap_data[LoggerColsEnum.PREDICTIONS],
            ]
        )

        self.plot_confusion_matrix(
            cf_labels,
            humps.camelize(LoggerColsEnum.PREDICTIONS),
            humps.camelize(LoggerColsEnum.REFERENCES),
            "Intents Confusion Matrix",
            out_dir / "intents_confusion_matrix.png",
        )


class RequestedSlotPredictionLogger(PredictionsLoggerBase):
    def __init__(self):
        self.preds = []
        self.refs = []
        self.is_correct = []

    def log(self, pred, ref, is_correct):
        self.preds.append(pred)
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path, top_k=15):
        data = self._get_stacked_bar_chart_data()


class BeliefGoalPredictionLogger(PredictionsLoggerBase):
    def __init__(self):
        self.refs = []
        self.is_correct = []

    def log(
        self,
        pred: Optional[SimpleTodBelief] = None,
        ref: Optional[SimpleTodBelief] = None,
        is_correct: bool = None,
    ):
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def _plot_graph(self, df: any, column: str, out_dir: str, top_k=15) -> None:
        top_data = (
            df[df[LoggerColsEnum.IS_CORRECT] == False]
            .groupby([column, LoggerColsEnum.IS_CORRECT])[column]
            .count()
            .reset_index(name=LoggerColsEnum.COUNT)
            .sort_values(LoggerColsEnum.COUNT, ascending=False)[:top_k]
        )
        data = (
            df[df[column].isin(top_data[column])]
            .groupby([column, LoggerColsEnum.IS_CORRECT])[column]
            .count()
            .reset_index(name=LoggerColsEnum.COUNT)
        )
        plt.style.use("ggplot")
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 10), dpi=150)
        sns.barplot(
            data=data,
            x=column,
            y=LoggerColsEnum.COUNT,
            hue=LoggerColsEnum.IS_CORRECT,
            palette=["r", "g"],
            saturation=0.5,
        )
        plt.xlabel(humps.pascalize(column))
        plt.ylabel(humps.pascalize(LoggerColsEnum.COUNT))
        plt.xticks(rotation=90, fontsize=8)
        plt.title(
            f"{humps.pascalize(column)} {humps.pascalize(LoggerColsEnum.PREDICTIONS)}"
        )
        plt.tight_layout()
        plt.legend(title=humps.pascalize(LoggerColsEnum.IS_CORRECT))
        plt.savefig(
            out_dir
            / f"Belief_{humps.pascalize(column)}_{LoggerColsEnum.PREDICTIONS}.png"
        )
        plt.close()

    def visualize(self, out_dir: Path):
        df = pd.concat(
            [
                pd.DataFrame(self.refs),
                pd.DataFrame(self.is_correct, columns=[LoggerColsEnum.IS_CORRECT]),
            ],
            axis=1,
        )
        [self._plot_graph(df, col, out_dir) for col in ["domain", "slot_name", "value"]]


class ActionGoalPredictionLogger(PredictionsLoggerBase):
    def __init__(self):
        self.preds = []
        self.refs = []
        self.is_correct = []

    def log(self, pred=None, ref=None, is_correct=None):
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path):
        return
        df = self._init_data_frame()
        df_false = self._get_false_predictions(df)
        error_refs = self._get_data_grouped_by_false_references(df_false)
