import abc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class PredictionsLoggerBase(abc.ABC):
    @abc.abstractmethod
    def log(self, pred, ref, is_correct):
        raise (NotImplementedError)

    @abc.abstractmethod
    def visualize(self, out_dir: Path):
        raise (NotImplementedError)


class IntentsPredictionLogger(PredictionsLoggerBase):
    def __init__(self):
        self.preds = []
        self.refs = []
        self.is_correct = []

    def log(self, pred, ref, is_correct):
        self.preds.append(pred)
        self.refs.append(ref)
        self.is_correct.append(is_correct)

    def visualize(self, out_dir: Path, top_k=7):
        plt.style.use("ggplot")
        sns.set(style="darkgrid")
        df = pd.DataFrame(
            {
                "predictions": self.preds,
                "references": self.refs,
                "is_correct": self.is_correct,
            }
        )
        df_correct = df[df["is_correct"] == True].iloc[:, :-1]
        df_incorrect = df[df["is_correct"] == False].iloc[:, :-1]

        heat_df = df_incorrect.pivot_table(
            index="predictions", columns="references", aggfunc="value_counts"
        ).head(top_k)
        plt.figure(figsize=(10, 10), dpi=150)
        heat = sns.heatmap(
            heat_df,
            cmap="rocket_r",
            annot=True,
            linewidths=1,
            linecolor="w",
            annot_kws={"fontsize": 8},
        )
        plt.yticks(fontsize=8)
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "intent_pair_errors.png")

        plt.figure()

        hist = sns.histplot(
            data=df_correct, x="references", label="Correct", color="green", alpha=0.5
        )
        hist = sns.histplot(
            data=df_incorrect, x="references", label="Errors", color="red", alpha=0.5
        )
        plt.xlabel("Reference Intents")
        plt.xticks(rotation=90, fontsize=8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "intent_ref_histogram.png")
