from pathlib import Path
import uuid
from dotmap import DotMap
import evaluate
import pandas as pd
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath("./src"))

from logger.results_logger import ResultsLogger
from metrics.bert_score_metric import BertScoreMetric


class CalcBertScore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.metric = BertScoreMetric()

    def run(self):
        if self.cfg.data:
            df = pd.DataFrame(self.cfg.data)
        else:
            if self.cfg.pred_path.is_absolute():
                csv_path = self.cfg.pred_path
            else:
                csv_path = self.cfg.project_root / self.cfg.pred_path

            df = pd.read_csv(csv_path)
        if self.cfg.out_path:
            out_path = self.cfg.out_path
        else:
            out_path = (
                self.cfg.project_root
                / self.cfg.out_root
                / (self.cfg.model_name + ".csv")
            )
        rl = ResultsLogger(self.cfg)
        df_cat = rl.get_data_by_settings(df)
        out = []

        cat_score, r_score, s_score = self.get_cat_score(df_cat)
        out.append(["all", cat_score, r_score, s_score])
        for category in tqdm(["seen", "unseen", "mixed"]):
            data = df_cat[df_cat.domain_category == category]
            if data.empty:
                continue
            cat_score, r_score, s_score = self.get_cat_score(data)
            out.append([category, cat_score, r_score, s_score])
        if self.cfg.out_root:
            self.cfg.out_root.mkdir(parents=True, exist_ok=True)

        out_df = pd.DataFrame(
            out, columns=["setting", "response", "retrieval", "slot_fill"]
        )
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        a = 1

    def get_cat_score(self, data):
        cat_score = self.get_score(data)
        retrieval_data = data[data.is_retrieval == 1]
        slot_fill_data = data[data.is_slot_fill == 1]
        r_score = self.get_score(retrieval_data)
        s_score = self.get_score(slot_fill_data)
        return cat_score, r_score, s_score

    def get_score(self, df):
        self.metric.reset()
        labels = df.label.to_list()
        preds = df.pred.to_list()
        if not labels or not preds:
            return 0
        self.metric._update(references=labels, predictions=preds)
        res = self.metric._compute()
        return round(res.f1, 4)


if __name__ == "__main__":
    dd = CalcBertScore(
        DotMap(
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            # raw_data_root="data/ketod",
            raw_data_root="data/dstc8-schema-guided-dialogue",
            out_root=Path("data_exploration/bert_scores"),
            pred_path=Path(
                # "/u/amo-d0/grad/adibm/data/download/sgd/llama_multi_sgd_all.csv"
                "/u/amo-d0/grad/adibm/data/download/autotod_ketod_all.csv"
            ),
            model_name="autotod_ketod",
        )
    )
    dd.run()
