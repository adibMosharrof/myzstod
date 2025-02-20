import itertools
from pathlib import Path
from dotmap import DotMap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class UserStudyAnalysis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.out_root = self.cfg.project_root / self.cfg.out_root
        self.cfg.out_root.mkdir(parents=True, exist_ok=True)
        self.task_map = {
            0: "Informativeness",
            1: "Fluency",
            2: "Task Completion",
        }
        self.model_names = {
            "soloist": "SOLOIST",
            "autotod": "AutoToD",
            "gpt": "GPT-2",
            "llama": "Llama 3.2",
            "flan": "Flan-T5",
        }
        self.model_names = {
            "soloist": "SOLOIST",
            "autotod": "AutoToD",
            "realTOD_claude": "Claude",
            "realTOD_gpt": "GPT-4o",
            "realTOD_llama": "Llama",
            "realTOD_deepseek": "DeepSeek",
        }

    def box_plot(self, data, task_name):
        data = data.drop("gt", axis=1)
        data.task = data.task.map(self.task_map)
        data = data.rename(columns=self.model_names)
        # models = data[self.model_names]
        model_names = list(self.model_names.values())
        df = data.melt(
            id_vars=["single_domain", "task", "dataset"],
            value_vars=list(model_names),
            var_name="Model",
            value_name="Score",
        )
        sns.set_style("whitegrid")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        # colors = sns.color_palette("Set2", 5)
        # colors = sns.color_palette("Set2", 6)

        colors = {
            model: color
            for model, color in zip(
                model_names,
                ["#A659C2", "#EE4266", "#3C85CF", "#0EAD69", "#FFD23F"],
                # ["green", "blue", "orange", "purple", "brown"],
            )
        }
        # sns.set_palette(colors)
        boxplot = sns.boxplot(
            hue="Model",
            y="Score",
            x="task",
            data=df,
            ax=ax,
            width=0.5,
            dodge=True,
            linewidth=1.2,
            # palette="colorblind6",
            hue_order=model_names,
            # palette=colors,
            palette="Set2",
        )
        # ax.set_title(f"Human Evaluation")
        ax.set_ylabel("")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.title("", fontsize=16, pad=15)
        plt.xlabel("", fontsize=10, labelpad=10)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=18)
        ax.set_yticks([1, 2, 3, 4, 5])
        # ax.set_xticks(ax.get_xticks() + 0.25)
        plt.tight_layout()

        plt.legend(
            title="",
            loc="upper center",
            fontsize=16,
            bbox_to_anchor=(0.5, -0.07),
            framealpha=0.5,
            frameon=False,
            # ncol=5,
            ncol=3,
            markerscale=2,
            # prop={"size": 15},
        )
        plt.subplots_adjust(bottom=0.14)
        plt.savefig(
            self.cfg.out_root / f"{task_name.lower().replace(' ', '_')}_scores.pdf",
            bbox_inches="tight",
        )

    def run(self):
        data = pd.read_csv(self.cfg.data_path)
        self.box_plot(data, "Overall")
        # for task in range(3):
        #     task_data = data[data["task"] == task]
        #     task_name = self.task_map[task]
        #     self.box_plot(task_data, task_name)


if __name__ == "__main__":
    usa = UserStudyAnalysis(
        DotMap(
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            # data_path="data/user_study/results/all.csv",
            # out_root="data_exploration/user_study/analysis/",
            data_path="data/user_study/results/realtod/real_tod_all.csv",
            out_root="data_exploration/user_study/analysis/realtod",
            # dataset_name="sgd",
            # dataset_name="bitod",
        )
    )

    usa.run()
