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

    def _get_false_predictions(self, df, group_column=[LoggerColsEnum.REFERENCES]):
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
            # .groupby(np.concatenate([group_columns, [LoggerColsEnum.IS_CORRECT]]))[
            .groupby([group_column, LoggerColsEnum.IS_CORRECT])[
                LoggerColsEnum.IS_CORRECT
            ]
            .count()
            .reset_index(name=LoggerColsEnum.COUNT)
        )

    def _get_stacked_bar_chart_data(
        self, group_column=LoggerColsEnum.REFERENCES, top_k=10, df=None
    ) -> StackedBarChartData:

        if df is None:
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

        df_false = self._get_false_predictions(df, group_column=group_column)
        error_refs = (
            df_false.groupby(group_column)[LoggerColsEnum.COUNT]
            .sum()
            .reset_index(name=LoggerColsEnum.COUNT)
            .sort_values(LoggerColsEnum.COUNT, ascending=False)
        )

        # stacked hbar plot
        top_error_refs_bar = error_refs[:top_k]
        df_bar = df[df[group_column].isin(top_error_refs_bar[group_column])]
        # sorting values by predictions where is_correct is false
        data_prop = pd.crosstab(
            index=df_bar[group_column],
            columns=df_bar[LoggerColsEnum.IS_CORRECT],
            normalize="index",
        ).sort_values(False)

        data_count = pd.crosstab(
            index=df_bar[group_column],
            columns=df_bar[LoggerColsEnum.IS_CORRECT],
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


class IntentsPredictionLogger(PredictionsLoggerBase):
    def __init__(self):
        super().__init__()

    def log(self, pred, ref, is_correct):
        self.preds.append(pred)
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path, top_k_bar=10, top_k_confusion=5):
        data = self._get_stacked_bar_chart_data(top_k=top_k_bar)

        self._plot_stacked_bar_chart(
            data,
            "Proportion",
            "Intents",
            "Intents Predictions",
            out_dir / "intent_predictions.png",
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
        super().__init__()

    def log(self, pred, ref, is_correct):
        self.preds.append(pred)
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path, top_k=10):
        data = self._get_stacked_bar_chart_data(top_k=top_k)
        self._plot_stacked_bar_chart(
            data,
            "Proportion",
            "Requested Slots",
            "Requested Slots Predictions",
            out_dir / "requested_slots_predictions.png",
        )


class GoalPredictionLogger(PredictionsLoggerBase):
    def __init__(self, step: GoalMetricConfigType = GoalMetricConfigType.BELIEF):
        super().__init__()
        self.step = step
        if step == GoalMetricConfigType.BELIEF:
            self.columns = ["domain", "slot_name", "value"]
        elif step == GoalMetricConfigType.ACTION:
            self.columns = ["domain", "action_type", "slot_name", "values"]

    def log(
        self,
        pred: Optional[SimpleTodBelief] = None,
        ref: Optional[SimpleTodBelief] = None,
        is_correct: bool = None,
    ):
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path):
        df = pd.concat(
            [
                pd.DataFrame(map(lambda x: x.__dict__, self.refs)),
                pd.DataFrame(self.is_correct, columns=[LoggerColsEnum.IS_CORRECT]),
            ],
            axis=1,
        )
        for col in self.columns:
            data = self._get_stacked_bar_chart_data(df=df, group_column=col)
            self._plot_stacked_bar_chart(
                data,
                "Proportion",
                f"{humps.pascalize(self.step.value)} {humps.pascalize(col)}s",
                f"{humps.pascalize(col)} Predictions",
                out_dir / f"{self.step}_{col}_predictions.png",
            )


class ActionGoalPredictionLogger(PredictionsLoggerBase):
    def __init__(self):
        super().__init__()

    def log(self, pred=None, ref=None, is_correct=None):
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path):
        df = pd.concat(
            [
                pd.DataFrame(map(lambda x: x.__dict__, self.refs)),
                pd.DataFrame(self.is_correct, columns=[LoggerColsEnum.IS_CORRECT]),
            ],
            axis=1,
        )
        for col in ["domain", "action_type", "slot_name", "values"]:
            data = self._get_stacked_bar_chart_data(df=df, group_column=col)
            self._plot_stacked_bar_chart(
                data,
                "Proportion",
                f"Action {humps.pascalize(col)}s",
                f"{humps.pascalize(col)} Predictions",
                out_dir / f"action_{col}_predictions.png",
            )
