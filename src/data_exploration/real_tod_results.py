import functools
from pathlib import Path
import uuid
from dotmap import DotMap
import evaluate
import pandas as pd
import os
import sys
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath("./src"))
from logger.results_logger import ResultsLogger
from metrics.bert_score_metric import BertScoreMetric


class RealTodResults:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.metric = BertScoreMetric()

    def run(self):
        out_path = (
            self.cfg.project_root
            / self.cfg.out_root
            / f"{self.cfg.dataset_name}_real_tod_results.csv"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = []
        for model_name, path in self.cfg.pred_paths:
            res = self.process_single_model(path, model_name)
            out.extend(res)
        out_df = pd.DataFrame(out)
        out_df.to_csv(out_path, index=False, encoding="utf-8")

    def process_single_model(self, path, model_name):
        pred_path = self.cfg.project_root / path
        df_csv = pd.read_csv(pred_path)
        df_csv = df_csv.sample(1000, random_state=42)
        rl = ResultsLogger(self.cfg)
        df_cat = rl.get_data_by_settings(df_csv)
        out = []

        self.setting_scores(model_name, df_cat, out, "all")
        df = df_cat[df_cat.domain_category == "unseen"]
        self.setting_scores(model_name, df, out, "unseen")
        return out

    def setting_scores(self, model_name, df, out, domain_setting):
        if df.empty:
            return
        (_, s_df), (_, m_df) = df.groupby(df["domains"].str.contains(","))
        # out.append(self.get_cat_score(df, model_name, "both"))
        # out.append(self.get_cat_score(s_df, model_name, "single"))
        # out.append(self.get_cat_score(m_df, model_name, "multi"))
        args_list = [
            (df, model_name, "both", domain_setting),
            (s_df, model_name, "single", domain_setting),
            (m_df, model_name, "multi", domain_setting),
        ]
        with ProcessPoolExecutor(max_workers=3) as executor:
            future_to_args = {
                executor.submit(self.get_cat_score, *args): args for args in args_list
            }
            for future in as_completed(future_to_args):
                result = future.result()
                out.append(result)

    def get_cat_score(self, df, model_name, setting, domain_setting):
        api_rows = df[df.turn_row_type == 1]
        out = DotMap()
        out.model_name = model_name
        out.domain_setting = domain_setting
        out.setting = setting
        out.api_method = api_rows["api_call_method"].mean().round(4)
        out.api_param_names = api_rows["api_call_param_names"].mean().round(4)
        out.api_param_values = api_rows["api_call_param_values"].mean().round(4)
        out.complete_api_call = api_rows["complete_api_call"].mean().round(4)
        relation_row = "api_call_param_relation"
        if relation_row in api_rows.columns:
            out[relation_row] = api_rows[relation_row].mean().round(4)

        response_rows = df[df.turn_row_type == 0]
        all_refs = []
        all_preds = []
        prefix = "system: "
        grouped = response_rows.sort_values(by=["dialog_id", "turn_id"]).groupby(
            "dialog_id"
        )
        for id, group in grouped:
            refs, preds = [], []
            for label, pred in tqdm(
                zip(group.label, group.pred), desc=f"setting:{setting}"
            ):
                refs.append(prefix + label)
                pred = "" if pd.isna(pred) else pred
                preds.append(prefix + pred)
            all_refs.append("".join(refs))
            all_preds.append("".join(preds))
        metric = BertScoreMetric()
        out.bert_score = self.get_score(all_refs, all_preds, metric)
        return dict(out)

    def get_score(self, labels, preds, metric):

        if not labels or not preds:
            return 0
        metric._update(references=labels, predictions=preds)
        res = metric._compute()
        return round(res.f1, 4)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    real_tod_results = RealTodResults(
        DotMap(
            project_root=Path("/u/amo-d0/grad/adibm/data/projects/ZSToD"),
            raw_data_root="data/dstc8-schema-guided-dialogue",
            dataset_name="sgd",
            # raw_data_root="data/bitod",
            # dataset_name="bitod",
            out_root="data_exploration/real_tod_results",
            pred_paths=[
                [
                    "zs_tod_sgd",
                    # "/u/amo-d0/grad/adibm/data/download/zstod_sgd_all.csv"
                    "/u/amo-d0/grad/adibm/data/download/new_results/zstod_sgd_all.csv",
                ],
                [
                    "auto_tod_sgd",
                    "/u/amo-d0/grad/adibm/data/download/autotod_sgd_all.csv",
                ],
                [
                    "soloist_sgd",
                    # "/u/amo-d0/grad/adibm/data/download/soloist_sgd_all.csv",
                    "/u/amo-d0/grad/adibm/data/download/new_results/soloist_sgd_all.csv",
                ],
                [
                    "simpletod_sgd",
                    # "/u/amo-d0/grad/adibm/data/download/simpletod_sgd_all.csv",
                    "/u/amo-d0/grad/adibm/data/download/new_results/simpletod_sgd_all.csv",
                ],
            ],
            # pred_paths=[
            #     [
            #         "zs_tod_bitod",
            #         "/u/amo-d0/grad/adibm/data/download/bitod/zstod_bitod_all.csv",
            #     ],
            #     [
            #         "auto_tod_bitod",
            #         "/u/amo-d0/grad/adibm/data/download/bitod/autotod_bitod_all.csv",
            #     ],
            #     [
            #         "soloist_bitod",
            #         "/u/amo-d0/grad/adibm/data/download/bitod/soloist_bitod_all.csv",
            #     ],
            #     [
            #         "simpletod_bitod",
            #         "/u/amo-d0/grad/adibm/data/download/bitod/simpletod_bitod_all.csv",
            #     ],
            # ],
            model_name="zstod_sgd",
        )
    )
    real_tod_results.run()
