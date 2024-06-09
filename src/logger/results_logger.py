from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotmap import DotMap
import hydra
from omegaconf import DictConfig
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))
from dstc.dstc_domains import DstcDomainBuilder, DstcDomains
from logger.inference_logger_dataclasses import ApiCallInferenceLogData
from my_enums import TurnRowType
import utils
import csv

# from pytablewriter import MarkdownTableWriter


class ResultsLogger:

    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.project_root = Path(cfg.project_root)
        self.cfg.raw_data_root = Path(cfg.raw_data_root)

    def get_csv(self, path):
        data = pd.read_csv(self.cfg.project_root / path)
        return data

    def get_group_metrics(self, group, metric_names):
        results = {}
        for metric_name in metric_names:
            try:
                metric = group[metric_name].mean().round(4)
                results[metric_name] = metric
            except AttributeError as e:
                print(f"no {metric_name} data for group")
        return results

    def get_regular_setting_results(self, results):
        csv_results = []
        rows_seen, rows_unseen, rows_mixed = self.get_data_by_settings(results)
        for rows, setting in zip(
            [results, rows_seen, rows_unseen, rows_mixed],
            ["all", "seen", "unseen", "mixed"],
        ):
            if len(rows) == 0:
                print(f"no data for setting {setting}")
                continue
            setting_results = {}
            setting_results["setting"] = setting
            response_rows = rows[rows["turn_row_type"] == TurnRowType.RESPONSE.value]
            setting_results.update(
                self.get_group_metrics(
                    response_rows, ["response_bleu", "response_gleu"]
                )
            )
            api_rows = rows[rows["turn_row_type"] == TurnRowType.API_CALL.value]
            setting_results.update(
                self.get_group_metrics(
                    api_rows,
                    [
                        "api_call_invoke",
                        "api_call_method",
                        "api_call_param_names",
                        "api_call_param_values",
                        "complete_api_call",
                    ],
                )
            )
            if 'ketod' in self.cfg.raw_data_root.name:
                ke_query_rows = rows[rows["turn_row_type"] == TurnRowType.KE_QUERY.value]
                setting_results.update(
                    self.get_group_metrics(
                        ke_query_rows,
                        [
                            "ke_api_call_invoke",
                            "ke_method",
                            "ke_params",
                            # "ke_param_values",
                            "complete_kb_call",
                        ],
                    )
                )

            retrieval_rows = rows[rows["is_retrieval"] == 1]
            retrieval_metrics = self.get_group_metrics(
                retrieval_rows, ["response_bleu", "response_gleu"]
            )
            setting_results.update(
                DotMap(
                    retrieval_bleu=retrieval_metrics["response_bleu"],
                    retrieval_gleu=retrieval_metrics["response_gleu"],
                )
            )

            slot_fill_rows = rows[rows["is_slot_fill"] == 1]
            slot_fill_metrics = self.get_group_metrics(
                slot_fill_rows, ["response_bleu", "response_gleu"]
            )
            setting_results.update(
                DotMap(
                    slot_fill_bleu=slot_fill_metrics["response_bleu"],
                    slot_fill_gleu=slot_fill_metrics["response_gleu"],
                )
            )
            csv_results.append(setting_results)
        keys = csv_results[0].keys()
        out_dir = Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "regular_results.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(csv_results)

    def get_data_by_settings(self, results):
        data_root = self.cfg.get("raw_data_root", None)
        if data_root is None:
            try:
                data_root = self.cfg.dataset["raw_data_root"]
            except KeyError as e:
                print("no data root in config")
                raise (e)

        db = DstcDomainBuilder(self.cfg.project_root / Path(data_root), 1)
        domain_map = {
            step: db.get_domains(step)
            for step in [DstcDomains.SEEN, DstcDomains.UNSEEN]
        }
        rows_by_domain_setting = {
            step: [] for step in [DstcDomains.SEEN, DstcDomains.UNSEEN]
        }
        mixed_domain_rows = []
        for i, row in results.iterrows():
            domains = row["domains"].split(",")
            in_seen_domains = []
            in_unseen_domains = []
            for dom in domains:
                if dom in domain_map[DstcDomains.SEEN]:
                    in_seen_domains.append(1)
                if dom in domain_map[DstcDomains.UNSEEN]:
                    in_unseen_domains.append(1)
            if len(in_seen_domains) > 0 and len(in_unseen_domains) > 0:
                mixed_domain_rows.append(i)
            elif len(in_seen_domains) > 0:
                rows_by_domain_setting[DstcDomains.SEEN].append(i)
            elif len(in_unseen_domains) > 0:
                rows_by_domain_setting[DstcDomains.UNSEEN].append(i)
            else:
                raise ValueError("no domain match")
        rows_seen = results.iloc[rows_by_domain_setting[DstcDomains.SEEN]]
        rows_unseen = results.iloc[rows_by_domain_setting[DstcDomains.UNSEEN]]
        rows_mixed = results.iloc[mixed_domain_rows]
        return rows_seen, rows_unseen, rows_mixed

    def get_results(self, results):
        response_rows = results[results["turn_row_type"] == TurnRowType.RESPONSE.value]
        domain_groups = response_rows.groupby("domains")
        # out = {}
        out = {item: {} for item in set(results["domains"])}
        for domain, group in domain_groups:
            bleu = group["response_bleu"].mean().round(4)
            gleu = group["response_gleu"].mean().round(4)
            out[domain].update(DotMap(response_bleu=bleu, response_gleu=gleu))

        api_rows = results[results["turn_row_type"] == TurnRowType.API_CALL.value]
        domain_groups = api_rows.groupby("domains")
        api_results = {}
        for domain, group in domain_groups:
            try:
                invoke = group["api_call_invoke"].mean().round(4)
                method = group["api_call_method"].mean().round(4)
                params = group["api_call_param_names"].mean().round(4)
                values = group["api_call_param_values"].mean().round(4)
                complete = group["complete_api_call"].mean().round(4)
                # api_results[domain] = DotMap(
                out[domain].update(
                    DotMap(
                        api_call_invoke=invoke,
                        api_call_method=method,
                        api_call_param_names=params,
                        api_call_param_values=values,
                        complete_api_call=complete,
                    )
                )
            except AttributeError as e:
                print("no api call data for domain ", domain)

        multi_domain_api_rows = results[results["is_multi_domain_api_call"] == 1]
        domain_groups = multi_domain_api_rows.groupby("domains")
        api_results = {}
        for domain, group in domain_groups:
            try:
                invoke = group["api_call_invoke"].mean().round(4)
                method = group["api_call_method"].mean().round(4)
                params = group["api_call_param_names"].mean().round(4)
                values = group["api_call_param_values"].mean().round(4)
                complete = group["complete_api_call"].mean().round(4)
                # api_results[domain] = DotMap(
                out[domain].update(
                    DotMap(
                        multi_api_call_invoke=invoke,
                        multi_api_call_method=method,
                        multi_api_call_param_names=params,
                        multi_api_call_param_values=values,
                        multi_complete_api_call=complete,
                    )
                )
            except AttributeError as e:
                print("no multi domain api call data for domain ", domain)

        query_rows = results[results["turn_row_type"] == TurnRowType.KE_QUERY.value]
        domain_groups = query_rows.groupby("domains")
        for domain, group in domain_groups:
            try:
                invoke = group["ke_api_call_invoke"].mean().round(4)
                method = group["ke_method"].mean().round(4)
                params = group["ke_params"].mean().round(4)
                complete = group["complete_kb_call"].mean().round(4)
                # api_results[domain] = DotMap(
                out[domain].update(
                    DotMap(
                        ke_api_call_invoke=invoke,
                        ke_method=method,
                        ke_params=params,
                        ke_param_values=values,
                        complete_kb_call=complete,
                    )
                )
            except AttributeError as e:
                print("no api call data for domain ", domain)

        retrieval_rows = results[results["is_retrieval"] == 1]
        domain_groups = retrieval_rows.groupby("domains")
        retrieval_results = {}
        for domain, group in domain_groups:
            try:
                bleu = group["response_bleu"].mean().round(4)
                gleu = group["response_gleu"].mean().round(4)
                # retrieval_results[domain] = DotMap(retrieval_bleu=bleu, retrieval_gleu=gleu)
                out[domain].update(DotMap(retrieval_bleu=bleu, retrieval_gleu=gleu))
            except AttributeError as e:
                print("no retrieval data for domain ", domain)

        slot_fill_rows = results[results["is_slot_fill"] == 1]
        domain_groups = slot_fill_rows.groupby("domains")
        slot_fill_results = {}
        for domain, group in domain_groups:
            try:
                bleu = group["response_bleu"].mean().round(4)
                gleu = group["response_gleu"].mean().round(4)
                # slot_fill_results[domain] = DotMap(slot_fill_bleu=bleu, slot_fill_gleu=gleu)
                out[domain].update(DotMap(slot_fill_bleu=bleu, slot_fill_gleu=gleu))
            except AttributeError as e:
                print("no slot fill data for domain ", domain)
        return out

    def write_results(self, results, chat_gpt_results):
        col_groups = DotMap(
            response=["response_bleu", "response_gleu"],
            api_call=[
                "api_call_invoke",
                "api_call_method",
                "api_call_param_names",
                "api_call_param_values",
                "complete_api_call",
            ],
            ke_api_call=[
                "ke_api_call_invoke",
                "ke_method",
                "ke_params",
                "ke_param_values",
                "complete_kb_call",
            ],
            multi_api_call=[
                "multi_api_call_invoke",
                "multi_api_call_method",
                "multi_api_call_param_names",
                "multi_api_call_param_values",
                "multi_complete_api_call",
            ],
            retrieval=["retrieval_bleu", "retrieval_gleu"],
            slot_fill=["slot_fill_bleu", "slot_fill_gleu"],
        )
        results = sorted(results.items())
        tables = []
        for key, col_group in col_groups.items():
            headers = ["domains"] + col_group * 2
            csv_path = Path(self.cfg.out_dir) / f"{key}.csv"
            rows = []
            for domain, item in results:
                row = [domain]
                for col in col_group:
                    r_value = item.get(col, None)
                    row.append(r_value)
                c_item = chat_gpt_results.get(domain, {})
                for col in col_group:
                    c_value = c_item.get(col, None)
                    row.append(c_value)
                rows.append(row)
            utils.write_csv(headers, rows, csv_path)

    def run(self):
        chat_gpt_csv = self.get_csv(self.cfg.chatgpt_results_path)
        results_csv = self.get_csv(self.cfg.results_path)
        setting_results = self.get_regular_setting_results(results_csv)
        turn_row_results = self.get_results(results_csv)
        chat_gpt_results = self.get_results(chat_gpt_csv)
        self.write_results(turn_row_results, chat_gpt_results)
        # with open(Path(os.getcwd()) / "results/combined.md", "w") as f:
        #     for table in md_tables:
        #         f.write(table)

        # return md_tables


@hydra.main(config_path="../../config/", config_name="results_logger")
def hydra_start(cfg: DictConfig) -> None:
    rl = ResultsLogger(cfg)
    rl.run()


if __name__ == "__main__":
    hydra_start()

# if __name__ == "__main__":

#     rl = ResultsLogger(
#         DotMap(
#             project_root="/mounts/u-amo-d1/adibm-data/projects/ZSToD/",
#             # results_path="playground/t5_tod_out/2024-04-28/16-44-39/results/all.csv",
#             results_path="results/all.csv",
#             chatgpt_results_path="data_exploration/chatgpt/chat_gpt_all.csv",
#             out_dir="results",
#         )
#     )
#     rl.run()
