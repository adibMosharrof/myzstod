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
from dstc.dstc_domains import DstcDomainBuilder, DstcDomains
from logger.inference_logger_dataclasses import ApiCallInferenceLogData
from metric_managers.nlg_api_call_metric_manager import NlgApiCallMetricManager
from prompts.nlg_prompt_manager import ChatGptPrompt
from logger.results_logger import ResultsLogger


from configs.dataprep_config import DataPrepConfig
from data_prep.data_prep_strategy_resolver import DataPrepStrategyResolver
from data_prep.dstc_base_data_prep import DstcBaseDataPrep
from my_enums import Steps, TurnRowType
import utils


from base_datamodule import SimpleTodDataSet
from tod.turns.zs_tod_turn import TodTurnApiCallCsvRow

# Set your OpenAI API key
openai.api_key = "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"


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

    def get_data_prep_class(self, cfg):
        dp_cfg = DataPrepConfig(**cfg)
        strategy = DataPrepStrategyResolver.resolve(dp_cfg)
        return DstcBaseDataPrep(dp_cfg, strategy)

    def prepare_data(self, stdp):
        stdp.run()

    def get_data_by_split_percent(self, data: list[any], split_percent: float):
        return data[: int(len(data) * split_percent)]

    def get_dataset(self):
        data_prep = self.get_data_prep_class(self.cfg)
        self.prepare_data(data_prep)
        csv_path = utils.get_csv_data_path(
            Steps.TEST.value,
            self.cfg.num_dialogs,
            cfg=data_prep.cfg,
        )
        try:
            data = utils.read_csv_dataclass(csv_path, self.tod_turn_row_cls)
        except FileNotFoundError:
            data = []

        return self.get_data_by_split_percent(data, self.cfg.data_split_percent)

    def get_prompts(self, schemas):
        data = self.get_dataset()
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

    def get_chatgpt_responses(self, client, item_prompts, nlg_metric_manager):
        if self.cfg.response_path:
            return pd.read_csv(self.cfg.project_root / self.cfg.response_path)
        outputs = []
        for row in item_prompts:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": row.prompt}],
            )
            out = ApiCallInferenceLogData(
                input_text=row.prompt,
                label=row.item.target,
                pred=response.choices[0].message.content,
                turn_row_type=row.item.turn_row_type,
                is_retrieval=row.item.is_retrieval,
                is_slot_fill=row.item.is_slot_fill,
                dialog_id=row.item.dialog_id,
                turn_id=row.item.turn_id,
                domains=row.item.domains_original,
            )
            outputs.append(out)
        nlg_metric_manager.data = outputs
        nlg_metric_manager.compute_row_wise_metrics()
        df = pd.DataFrame(nlg_metric_manager.data)
        csv_root = os.getcwd() / Path(self.cfg.out_dir)
        csv_root.mkdir(parents=True, exist_ok=True)
        csv_path = csv_root / "chatgpt_inference.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        return df
        # return csv_path

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
            setting_slot_fill_rows = pd.DataFrame()
            setting_retrieval_rows = pd.DataFrame()
            for domain in domains:
                df = responses[responses.domains == domain]
                response_rows = df[df.turn_row_type == 0]
                api_rows = df[df.turn_row_type == 1]
                multi_dom_api_rows = df[
                    df.turn_row_type == 1 and df.is_multi_domain_api_call == 1
                ]
                slot_fill_rows = df[df.is_slot_fill == 1]
                retrieval_rows = df[df.is_retrieval == 1]
                setting_response_rows = pd.concat(
                    [setting_response_rows, response_rows]
                )
                setting_api_rows = pd.concat([setting_api_rows, api_rows])
                setting_multi_dom_api_rows = pd.concat(
                    [setting_multi_dom_api_rows, multi_dom_api_rows]
                )
                setting_slot_fill_rows = pd.concat(
                    [setting_slot_fill_rows, slot_fill_rows]
                )
                setting_retrieval_rows = pd.concat(
                    [setting_retrieval_rows, retrieval_rows]
                )
            res = {}
            res["response_bleu"] = setting_response_rows.response_bleu.mean().round(4)
            # res["response_gleu"] = setting_response_rows.response_gleu.mean()
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

            res["slot_fill"] = setting_slot_fill_rows.response_bleu.mean().round(4)
            res["retrieval"] = retrieval_rows.response_bleu.mean().round(4)
            results[setting] = res

        rl = ResultsLogger(self.cfg)
        out = rl.get_results(responses)
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
        # results_df = pd.DataFrame(results)
        # results_df.to_csv(
        #     out_dir / "results.csv",
        #     index=False,
        #     encoding="utf-8",
        # )
        results_df = pd.DataFrame(
            [
                {"domain": "all", **results["all"]},
                {"domain": "seen", **results["seen"]},
                {"domain": "unseen", **results["unseen"]},
            ]
        )
        results_df.to_csv(
            out_dir / "regular_results.csv", index=False, encoding="utf-8"
        )
        # results_seen = pd.DataFrame(results["seen"])
        # results_unseen = pd.DataFrame(results["unseen"])
        # results_all = pd.DataFrame(results["all"])
        # results_seen.to_csv(
        #     out_dir / "results_seen.csv",
        #     index=False,
        #     encoding="utf-8",
        # )
        # results_unseen.to_csv(
        #     out_dir / "results_unseen.csv",
        #     index=False,
        #     encoding="utf-8",
        # )
        # results_all.to_csv(
        #     out_dir / "results_all.csv",
        #     index=False,
        #     encoding="utf-8",
        # )
        a = 1

    def run(self):
        client = openai.OpenAI(
            api_key="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"
        )
        steps = Steps.list()
        schemas = {}
        for d in [get_schemas(self.cfg.raw_data_root, step) for step in steps]:
            schemas.update(d)
        item_prompts = self.get_prompts(schemas)

        nlg_metric_manager = NlgApiCallMetricManager(self.logger)
        responses = self.get_chatgpt_responses(client, item_prompts, nlg_metric_manager)
        self.get_metrics(responses, nlg_metric_manager)


@hydra.main(config_path="../config/inference/", config_name="chatgpt_inference")
def hydra_start(cfg: DictConfig) -> None:
    cgi = ChatGptInference(cfg)
    cgi.run()


if __name__ == "__main__":
    hydra_start()
