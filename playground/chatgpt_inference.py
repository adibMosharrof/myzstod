import itertools
import logging
from pathlib import Path
from dotmap import DotMap
import hydra
from omegaconf import DictConfig
import openai
import pandas as pd
from sgd_dstc8_data_model.dstc_dataclasses import get_schemas
import os
import sys

sys.path.insert(0, os.path.abspath("./src"))
from metric_managers.metric_manager_factory import MetricManagerFactory
from t5_datamodule import T5DataModule

from dstc.dstc_domains import DstcDomainBuilder, DstcDomains
from logger.inference_logger_dataclasses import (
    ApiCallInferenceLogData,
    KetodInferenceLogData,
)
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from prompts.nlg_prompt_manager import ChatGptPrompt
from logger.results_logger import ResultsLogger
from tqdm import tqdm
import numpy as np
from configs.dataprep_config import DataPrepConfig
from data_prep.data_prep_strategy_resolver import DataPrepStrategyResolver
from data_prep.dstc_base_data_prep import DstcBaseDataPrep
from my_enums import Steps, TurnRowType
import utils
import data_prep.data_prep_utils as data_prep_utils
from pathos.multiprocessing import ProcessingPool as Pool
from base_datamodule import SimpleTodDataSet
from tod.turns.zs_tod_turn import TodTurnApiCallCsvRow

# Set your OpenAI API key
openai.api_key = ""


class ChatGptInference:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.cfg.project_root = Path(self.cfg.project_root)
        self.cfg.raw_data_root = self.cfg.project_root / self.cfg.raw_data_root
        self.tod_turn_row_cls = TodTurnApiCallCsvRow
        self.prompt_cls = ChatGptPrompt()
        formatter = logging.Formatter(fmt="%(message)s")
        root_logger = logging.getLogger()  # no name
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(formatter)
        self.logger = root_logger
        self.metric_manager = NlgApiCallMetricManager(self.logger)

    def get_prompts(self, schemas):
        dm = T5DataModule(self.cfg, None, schemas)
        train_dataset, val_dataset, test_datasets = dm.load_data()
        data = test_datasets[0].data
        all_prompts = []
        for item in data:
            prompt = self.prompt_cls.get_prompt(
                item.domains,
                item.schema,
                item.context,
                all_schema=schemas,
                domains_original=item.domains_original,
            )
            all_prompts.append(DotMap(prompt=prompt, item=item))
        return all_prompts

    def query_chatgpt(self, row):
        client = openai.OpenAI(
            api_key="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"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": row.prompt}],
        )
        out = KetodInferenceLogData(
            input_text=row.prompt,
            label=row.item.target,
            pred=response.choices[0].message.content,
            turn_row_type=row.item.turn_row_type,
            is_retrieval=row.item.is_retrieval,
            is_slot_fill=row.item.is_slot_fill,
            dialog_id=row.item.dialog_id,
            turn_id=row.item.turn_id,
            is_multi_domain_api_call=row.item.is_multi_domain_api_call,
            domains=row.item.domains_original,
            complete_kb_call=row.get("complete_kb_call", None),
            ke_method=row.get("ke_method", None),
            ke_params=row.get("ke_params", None),
            ke_api_call_invoke=row.get("ke_api_call_invoke", None),
        )
        return out

    def get_chatgpt_responses(self, item_prompts, nlg_metric_manager):
        if self.cfg.response_path:
            return pd.read_csv(self.cfg.project_root / self.cfg.response_path)
        outputs = []
        if self.cfg.is_multi_process:
            outputs = list(
                tqdm(
                    Pool().imap(
                        self.query_chatgpt,
                        item_prompts,
                    ),
                    total=len(item_prompts),
                )
            )
        else:
            outputs = [self.query_chatgpt(row) for row in item_prompts]
        nlg_metric_manager.data = outputs
        nlg_metric_manager.compute_row_wise_metrics()
        df = pd.DataFrame(nlg_metric_manager.data)
        csv_root = os.getcwd() / Path(self.cfg.out_dir)
        csv_root.mkdir(parents=True, exist_ok=True)
        csv_path = csv_root / "chatgpt_inference.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df

    def get_metrics(self, responses, nlg_metric_manager):
        domain_builder = DstcDomainBuilder(self.cfg.raw_data_root, 1)
        regular_domains = {
            setting: domain_builder.get_domains(setting)
            for setting in DstcDomains.regular_settings()
        }
        results = {}
        for setting, domains in regular_domains.items():
            setting_response_rows = pd.DataFrame()
            setting_api_rows = pd.DataFrame()
            setting_multi_dom_api_rows = pd.DataFrame()
            setting_ke_rows = pd.DataFrame()
            # setting_multi_dom_ke_rows = pd.DataFrame()
            setting_slot_fill_rows = pd.DataFrame()
            setting_retrieval_rows = pd.DataFrame()
            domain_rows = []
            for i, row in responses.iterrows():
                if data_prep_utils.is_dialogue_in_domain(
                    row.domains.split(","), domains
                ):
                    domain_rows.append(row)
            if len(domain_rows) == 0:
                print(f"no data for setting {setting}")
                continue

            df = pd.DataFrame(domain_rows)
            response_rows = df[df.turn_row_type == 0]
            api_rows = df[df.turn_row_type == 1]
            ke_rows = df[df.turn_row_type == 2]
            # multi_dom_api_rows = df[
            #     df.turn_row_type == 1 and df.is_multi_domain_api_call == 1
            # ]
            # multi_dom_api_rows = df.query(
            #     "turn_row_type ==1 & is_multi_domain_api_call == 1"
            # )
            multi_dom_api_rows = api_rows[api_rows.is_multi_domain_api_call == 1]
            multi_dom_ke_rows = ke_rows[ke_rows.is_multi_domain_api_call == 1]
            slot_fill_rows = df[df.is_slot_fill == 1]
            retrieval_rows = df[df.is_retrieval == 1]
            setting_response_rows = pd.concat([setting_response_rows, response_rows])
            setting_api_rows = pd.concat([setting_api_rows, api_rows])
            setting_multi_dom_api_rows = pd.concat(
                [setting_multi_dom_api_rows, multi_dom_api_rows]
            )
            setting_ke_rows = pd.concat([setting_ke_rows, ke_rows])
            # setting_multi_dom_ke_rows = pd.concat(
            #     [setting_multi_dom_ke_rows, multi_dom_ke_rows]
            # )
            setting_slot_fill_rows = pd.concat([setting_slot_fill_rows, slot_fill_rows])
            setting_retrieval_rows = pd.concat([setting_retrieval_rows, retrieval_rows])
            res = {}
            res["response_bleu"] = setting_response_rows.response_bleu.mean().round(4)
            res["response_gleu"] = setting_response_rows.response_gleu.mean().round(4)
            res["complete_api_call"] = setting_api_rows.complete_api_call.mean().round(
                4
            )
            res["api_call_invoke"] = setting_api_rows.api_call_invoke.mean().round(4)
            res["api_call_method"] = setting_api_rows.api_call_method.mean().round(4)
            res["api_call_param_names"] = (
                setting_api_rows.api_call_param_names.mean().round(4)
            )
            res["api_call_param_values"] = (
                setting_api_rows.api_call_param_values.mean().round(4)
            )

            if len(setting_multi_dom_api_rows) > 0:
                res["multi_api_call_invoke"] = (
                    setting_multi_dom_api_rows.api_call_invoke.mean().round(4)
                )
                res["multi_api_call_method"] = (
                    setting_multi_dom_api_rows.api_call_method.mean().round(4)
                )
                res["multi_api_call_param_names"] = (
                    setting_multi_dom_api_rows.api_call_param_names.mean().round(4)
                )
                res["multi_api_call_param_values"] = (
                    setting_multi_dom_api_rows.api_call_param_values.mean().round(4)
                )
                res["multi_complete_api_call"] = (
                    setting_multi_dom_api_rows.complete_api_call.mean().round(4)
                )
            else:
                res["multi_api_call_invoke"] = 0
                res["multi_api_call_method"] = 0
                res["multi_api_call_param_names"] = 0
                res["multi_api_call_param_values"] = 0
                res["multi_complete_api_call"] = 0
            if "ketod" in self.cfg.raw_data_root.name:
                res["ke_api_call_invoke"] = (
                    setting_ke_rows.ke_api_call_invoke.mean().round(4)
                )
                res["ke_method"] = setting_ke_rows.ke_method.mean().round(4)
                res["ke_params"] = setting_ke_rows.ke_params.mean().round(4)
                res["ke_param_values"] = setting_ke_rows.ke_param_values.mean().round(4)

            # if len(setting_multi_dom_ke_rows) > 0:
            #     res["multi_domain_ke_api_call_invoke"] = (
            #         setting_multi_dom_ke_rows.api_call_invoke.mean().round(4)
            #     )
            #     res["multi_domain_ke_method"] = (
            #         setting_multi_dom_ke_rows.api_call_method.mean().round(4)
            #     )
            #     res["multi_domain_ke_params"] = (
            #         setting_multi_dom_ke_rows.api_call_param_names.mean().round(4)
            #     )
            #     res["multi_domain_ke_param_values"] = (
            #         setting_multi_dom_ke_rows.api_call_param_values.mean().round(4)
            #     )
            #     res["multi_domain_ke_complete_api_call"] = (
            #         setting_multi_dom_ke_rows.complete_api_call.mean().round(4)
            #     )
            # else:
            #     res["multi_domain_ke_api_call_invoke"] = 0
            #     res["multi_domain_ke_method"] = 0
            #     res["multi_domain_ke_params"] = 0
            #     res["multi_domain_ke_param_values"] = 0
            #     res["multi_domain_ke_complete_api_call"] = 0

            if len(setting_slot_fill_rows) > 0:
                res["slot_fill_bleu"] = (
                    setting_slot_fill_rows.response_bleu.mean().round(4)
                )
                res["slot_fill_gleu"] = (
                    setting_slot_fill_rows.response_gleu.mean().round(4)
                )
            else:
                res["slot_fill_bleu"] = 0
                res["slot_fill_gleu"] = 0
            if len(retrieval_rows) > 0:
                res["retrieval_bleu"] = retrieval_rows.response_bleu.mean().round(4)
                res["retrieval_gleu"] = retrieval_rows.response_gleu.mean().round(4)
            else:
                res["retrieval_bleu"] = 0
                res["retrieval_gleu"] = 0
            results[setting] = res

        rl = ResultsLogger(self.cfg)
        responses_with_dom_category = rl.get_data_by_settings(responses)
        out = rl.get_results(responses_with_dom_category)
        domain_data = [
            {
                "domain": dom_name,
                **metrics,
            }
            for dom_name, metrics in sorted(out.items())
        ]
        domain_wise_df = pd.DataFrame(domain_data)
        out_dir = os.getcwd() / Path(self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        domain_wise_df.to_csv(
            out_dir / "domain_wise_metrics.csv",
            index=False,
            encoding="utf-8",
        )
        rl.get_regular_setting_results(responses_with_dom_category)
        # results_dict = []
        # for domain, metrics in results.items():
        #     results_dict.append({"domain": domain, **metrics})
        # results_df = pd.DataFrame(results_dict)

        # results_df.to_csv(
        #     out_dir / "regular_results.csv", index=False, encoding="utf-8"
        # )

    def run(self):
        steps = Steps.list()
        schemas = {}
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in steps]:
            schemas.update(d)
        item_prompts = self.get_prompts(schemas)

        metric_manager = MetricManagerFactory.get_metric_manager(
            self.cfg.context_type, None, self.logger
        )
        responses = self.get_chatgpt_responses(item_prompts, metric_manager)
        self.get_metrics(responses, metric_manager)


@hydra.main(config_path="../config/inference/", config_name="chatgpt_inference")
def hydra_start(cfg: DictConfig) -> None:
    cgi = ChatGptInference(cfg)
    cgi.run()


if __name__ == "__main__":
    hydra_start()
