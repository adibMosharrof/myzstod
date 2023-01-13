import os
import sys
sys.path.insert(0, os.path.abspath("./src"))

from contrastive.contrastive_datamodule import ContrastiveDataModule
from pathlib import Path
from my_enums import Steps

from dotmap import DotMap
from umap import UMAP
import plotly.express as px
import utils
import pandas as pd
from torch.utils.data import DataLoader

class ContrastViz:
    def __init__(self, config):
        self.config = config
        # csv_data = pd.read_csv(self.config.processed_data_root / self.config.csv_file_name)
        self.dm = ContrastiveDataModule(
            dict(config)
        )
        a=1



    def run(self):
        umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap.fit(self.data)
        umap_data = umap.transform(self.data)
        umap_data = pd.DataFrame(umap_data, columns=["x", "y"])
        umap_data["label"] = self.data["label"]
        fig = px.scatter(umap_data, x="x", y="y", color="label")
        fig.show()

if __name__ == "__main__":
    cv = ContrastViz(
        DotMap(
            # processed_data_root=Path("processed_data/simple_tod")/ Steps.TRAIN.value,
            num_dialogs=[1, 1, 1],
            # csv_file_name="v0_context_type_short_repr_multi_head_False_multi_task_False_1_1_1_schema_True_user_actions_True_sys_actions_True_turns_26_service_results_True_dialogs_1_delexicalize_False_domain_setting_ALL_train_domains_1.0.csv",
        )
    )
    cv.run()
