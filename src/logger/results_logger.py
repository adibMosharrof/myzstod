from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotmap import DotMap
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))
from logger.inference_logger_dataclasses import ApiCallInferenceLogData
from my_enums import TurnRowType
import utils

# from pytablewriter import MarkdownTableWriter


class ResultsLogger:

    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.project_root = Path(cfg.project_root)

    def get_csv(self, path):
        # return pd.read_csv(self.cfg.project_root / path)
        data = pd.read_csv(self.cfg.project_root / path)
        return data

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
                        api_call_invoke=invoke,
                        api_call_method=method,
                        api_call_param_names=params,
                        api_call_param_values=values,
                        complete_api_call=complete,
                    )
                )
            except AttributeError as e:
                print("no multi domain api call data for domain ", domain)

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
        a = 1

    def write_results(self, results, chat_gpt_results):
        col_groups = DotMap(
            response=["response_bleu", "response_gleu"],
            api_call=[
                "api_call_invoke",
                "api_call_method",
                "api_call_param_names",
                "api_call_param_values",
            ],
            retrieval=["retrieval_bleu", "retrieval_gleu"],
            slot_fill=["slot_fill_bleu", "slot_fill_gleu"],
        )
        results = sorted(results.items())
        tables = []
        # chat_gpt_results = sorted(chat_gpt_results.items())
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
            # writer = MarkdownTableWriter(
            #     table_name=table_name,
            #     headers=["domain"] + col_group * 2,
            #     value_matrix=rows,
            # )
            # writer.write_table()
            # tables.append(writer.dumps())
        # return tables
        a = 1

    def run(self):
        chat_gpt_csv = self.get_csv(self.cfg.chatgpt_results_path)
        results_csv = self.get_csv(self.cfg.results_path)
        turn_row_results = self.get_results(results_csv)
        chat_gpt_results = self.get_results(chat_gpt_csv)
        self.write_results(turn_row_results, chat_gpt_results)
        # with open(Path(os.getcwd()) / "results/combined.md", "w") as f:
        #     for table in md_tables:
        #         f.write(table)

        # return md_tables


if __name__ == "__main__":

    rl = ResultsLogger(
        DotMap(
            project_root="/mounts/u-amo-d1/adibm-data/projects/ZSToD/",
            # results_path="playground/t5_tod_out/2024-04-28/16-44-39/results/all.csv",
            results_path="playground/t5_tod_out/2024-05-23/06-16-46/results/Buses_3,RentalCars_3.csv",
            chatgpt_results_path="playground/t5_tod_out/2024-05-15/02-29-21/results/chatgpt_inference.csv",
            out_dir="results",
        )
    )
    rl.run()
