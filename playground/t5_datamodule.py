from dotmap import DotMap
import numpy as np
import torch
from base_datamodule import SimpleTodDataSet, TodTrainRowCollator
from dstc.dstc_domains import DstcDomainBuilder
from my_enums import Steps
from prompts.nlg_prompt_manager import NlgPromptFactory
from prompts.prompt_constants import NlgPromptType
from tod.turns.zs_tod_turn import (
    TodTurnCsvRow,
    TodTurnCsvRowFactory,
    TodTurnApiCallCsvRow,
)
from tod_datamodules import TodDataModule
import utils
from utilities import text_utilities
from configs.dm_config import DataModuleConfig


class T5DataModule:
    def __init__(self, cfg, tokenizer, schemas):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.schemas = schemas
        self.nlg_prompt_cls = NlgPromptFactory.get_handler(
            cfg.prompt_type, cfg.context_type
        )
        self.domain_builder = DstcDomainBuilder(
            self.cfg.raw_data_root, self.cfg.data_split_percent[0]
        )
        self.train_domains = self.domain_builder.get_domains(
            self.cfg.train_domain_settings
        )

    def my_tokenize(self, text: str, max_len: int = None):
        tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=max_len)
        return tokens.to(dtype=torch.int32)[0]

    def trim_dialog_history(
        self,
        item: TodTurnCsvRow,
        trim_len: int,
        other_domain: str,
        other_domain_schema: str,
    ):
        dialog_history_tokens = self.my_tokenize(item.context)
        trimmed_history_tokens = dialog_history_tokens[trim_len + 15 :]
        trimmed_history_text = self.tokenizer.decode(trimmed_history_tokens)
        context_text = self.nlg_prompt_cls.get_prompt(
            item.domains,
            item.schema,
            trimmed_history_text,
            other_domain,
            other_domain_schema,
        )
        context_tokens = self.my_tokenize(context_text)
        if len(context_tokens) > self.cfg.test_prompt_max_len:
            overflow_tokens = len(context_tokens) - self.cfg.test_prompt_max_len
            return self.trim_dialog_history(
                item, -overflow_tokens, other_domain, other_domain_schema
            )
        return context_tokens

    def get_other_domain(self, item):
        domain = item.domains_original
        filtered_domains = [d for d in self.train_domains if d != domain]
        other_domain = np.random.choice(filtered_domains)
        other_domain_schema = self.schemas[other_domain]
        return (
            text_utilities.get_nlg_service_name(other_domain),
            other_domain_schema.get_nlg_repr(),
        )

    def collate_single_item(
        self, item: TodTurnCsvRow, target_max_len: int
    ) -> TodTrainRowCollator:
        other_domain, other_domain_schema = None, None
        if self.cfg.prompt_type == NlgPromptType.MULTI_DOMAIN.value:
            other_domain, other_domain_schema = self.get_other_domain(item)
        context_text = self.nlg_prompt_cls.get_prompt(
            item.domains,
            item.schema,
            item.context,
            other_domain,
            other_domain_schema,
        )
        context_tokens = self.my_tokenize(context_text)
        context_unused_len = self.cfg.test_prompt_max_len - len(context_tokens)
        if context_unused_len < 0:
            context_tokens = self.trim_dialog_history(
                item, -context_unused_len, other_domain, other_domain_schema
            )
            context_unused_len = self.cfg.test_prompt_max_len - len(context_tokens)
        pad = torch.full([context_unused_len], self.tokenizer.pad_token_id)
        input_tokens = torch.cat(
            [
                pad,
                context_tokens,
            ]
        )
        attention_mask = input_tokens.ne(self.tokenizer.pad_token_id).to(torch.int32)

        target_tokens = self.my_tokenize(item.target)
        target_unused_len = target_max_len - len(target_tokens)
        if target_unused_len < 0:
            raise Exception("Target is too long")
        label = torch.cat(
            [
                target_tokens,
                torch.full([target_unused_len], self.tokenizer.pad_token_id),
                torch.full([1], self.tokenizer.eos_token_id),
            ]
        )
        return TodTrainRowCollator(input_tokens, label, attention_mask)

    def tod_train_collate(self, batch: list[TodTurnCsvRow]):
        all_input_tokens = []
        all_labels = []
        all_attention_masks = []

        target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
        for item in batch:
            row = self.collate_single_item(item, target_max_len)
            all_input_tokens.append(row.input_tokens)
            all_attention_masks.append(row.attention_mask)
            all_labels.append(row.label)
        return DotMap(
            {
                "input_ids": torch.stack(all_input_tokens),
                "labels": torch.stack(all_labels),
                "attention_mask": torch.stack(all_attention_masks),
            }
        )

    def tod_test_collate(self, batch: list[TodTurnApiCallCsvRow]):
        all_input_tokens = []
        all_labels = []
        all_attention_masks = []
        all_turn_row_type = []

        target_max_len = self.cfg.max_token_len - self.cfg.test_prompt_max_len
        for item in batch:
            row = self.collate_single_item(item, target_max_len)
            all_input_tokens.append(row.input_tokens)
            all_attention_masks.append(row.attention_mask)
            all_labels.append(row.label)
            all_turn_row_type.append(torch.tensor(item.turn_row_type))
        return DotMap(
            {
                "input_ids": torch.stack(all_input_tokens),
                "labels": torch.stack(all_labels),
                "attention_masks": torch.stack(all_attention_masks),
                "turn_row_types": torch.stack(all_turn_row_type),
            }
        )

    def get_data_by_split_percent(
        self, data: list[TodTurnCsvRow], split_percent: float
    ):
        return data[: int(len(data) * split_percent)]

    def get_dms(self):
        steps = Steps.list()

        csv_row_cls = TodTurnCsvRowFactory.get_handler(self.cfg)
        return [
            TodDataModule(
                DataModuleConfig(tokenizer=self.tokenizer, **self.cfg),
                steps=steps,
                tod_turn_row_cls=csv_row_cls,
            )
        ]

    def load_data_from_files(self):
        train_fp = (
            self.cfg.project_root
            / "playground"
            / "data"
            / "train"
            / self.cfg.train_csv_file
        )
        val_fp = (
            self.cfg.project_root
            / "playground"
            / "data"
            / "dev"
            / self.cfg.dev_csv_file
        )
        test_fp = (
            self.cfg.project_root
            / "playground"
            / "data"
            / "test"
            / self.cfg.test_csv_file
        )
        train_data = utils.read_csv_dataclass(train_fp, TodTurnCsvRow)
        val_data = utils.read_csv_dataclass(val_fp, TodTurnCsvRow)
        test_data = utils.read_csv_dataclass(test_fp, TodTurnCsvRow)
        datasets = [
            SimpleTodDataSet(self.get_data_by_split_percent(data, split))
            for data, split in zip(
                [train_data, val_data, test_data], self.cfg.data_split_percent
            )
        ]
        return (*datasets,)

    def load_data(self):
        tod_dms = self.get_dms()[0].datasets
        return tod_dms["train"], tod_dms["dev"], tod_dms["test"]
        fp = self.cfg.project_root / "playground" / "data" / self.cfg.csv_file
        data = utils.read_csv_dataclass(fp, TodTurnCsvRow)
        df = pd.DataFrame([vars(d) for d in data])
        # df = pd.read_csv(fp, encoding="ISO-8859-1", header=None)

        df = df.sample(1200, random_state=420)

        # divide into test and train
        train_df = df.sample(frac=0.8, random_state=420)
        rest_df = df.drop(train_df.index)
        val_df = rest_df.sample(frac=0.5, random_state=420)
        test_df = rest_df.drop(val_df.index)
        datasets = [
            SimpleTodDataSet(
                df.apply(lambda row: TodTurnCsvRow(**row), axis=1).to_list()
            )
            for df in [train_df, val_df, test_df]
        ]
        return (*datasets,)
