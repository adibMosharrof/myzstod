from __future__ import annotations

from pathlib import Path
import re
from transformers import AutoTokenizer, GPT2LMHeadModel

from generation.generation_base import GenerationBase
from generation.multi_head_generation import MultiHeadGeneration
from generation.simple_generation import SimpleGeneration

from multi_head.mh_dataclasses import MultiHeadDictFactory
from multi_head.mh_datamodule import MultiLMHeadDatamodule
from multi_head.mh_model import GPT2MultiLMHeadModel
from my_enums import ContextType, SpecialTokens
from simple_tod_dataclasses import TodTestDataBatch
from tod_datamodules import TodDataModule
import utils
import dstc.dstc_utils as dstc_utils
from typing import TYPE_CHECKING
from configs.dm_config import DataModuleConfig

if TYPE_CHECKING:
    from configs.trainer_config import TrainerConfig
    from base_datamodule import BaseDataModule

class InferenceConfig:
    def __init__(
        self,
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 100,
        max_token_len: int = 512,
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        project_root: str = "/mounts/u-amo-d0/grad/adibm/data/projects/ZSToD",
        data_prep_out_root: str = "processed_data/simple_tod",
        predictions_log_dir: str = "predictions_logs",
        num_test_dialogs: int = 17,
        delexicalize: bool = False,
        model: str = "outputs/2022-07-26/22-28-09/results/train/checkpoint-7067",
        model_name: str = "gpt2",
        generate_max_len: int = 1024,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_percentage: float = 1.0,
        test_domain_settings: list[str] = None,
        out_dir: str = "results",
        tokenizer: AutoTokenizer = None,
        test_prompt_max_len: int = 799,
        is_multi_task: bool = False,
        is_multi_head: bool = False,
        is_multi_decoder: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_user_actions: bool = False,
        should_add_sys_actions: bool = False,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        postprocess_generation: bool = True,
        mh_fact: MultiHeadDictFactory = None,
        data_prep_multi_process: bool = True,
        wandb: any = None,
        datamodule:'BaseDataModule' = None
    ) -> None:
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.max_token_len = max_token_len
        self.project_root = Path(project_root)
        self.raw_data_root = self.project_root / raw_data_root
        self.data_prep_out_root = data_prep_out_root
        self.num_test_dialogs = num_test_dialogs
        self.delexicalize = delexicalize
        self.is_multi_head = is_multi_head
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer else self._get_tokenizer(model)
        self.mh_fact = (
            mh_fact
            if mh_fact
            else MultiHeadDictFactory(self.tokenizer)
            if is_multi_head
            else None
        )
        self.model = self._get_model(model)
        self.generate_max_len = generate_max_len
        self.train_domain_percentage = train_domain_percentage
        self.test_domain_settings = test_domain_settings or ["all", "seen", "unseen"]
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.out_dir = out_dir
        self.test_prompt_max_len = test_prompt_max_len
        self.predictions_log_dir = Path(predictions_log_dir)
        self.predictions_log_dir.mkdir(parents=True, exist_ok=True)
        self.is_multi_task = is_multi_task
        self.multi_tasks = (
            multi_tasks if self.is_multi_task and multi_tasks else [1, 1, 1]
        )
        self.is_multi_decoder = is_multi_decoder
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        self.logger = utils.get_logger()
        self.padding_regexp = re.compile(re.escape(SpecialTokens.pad_token))
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results
        self.postprocess_generation = postprocess_generation

        self.generation_handler: GenerationBase = (
            MultiHeadGeneration(self.model, self.tokenizer)
            if is_multi_head
            else SimpleGeneration(self.model, self.tokenizer)
        )
        # self.contrastive_model = contrastive_model
        self.data_prep_multi_process = data_prep_multi_process
        self.wandb = wandb
        self.datamodule = datamodule or self._get_datamodule()

    def _get_tokenizer(self, model_path_str: str):
        model_path: Path = self.project_root / model_path_str
        try:
            # with specifig checkpoint number (results/train/checkpoint-1000)
            tok_path = model_path.parent.parent.parent / "tokenizer"
            # checkpoint not provided (results/train)
            if not tok_path.exists():
                tok_path = model_path.parent.parent / "tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except OSError:
            self.logger.info(
                'Could not find tokenizer for model "{}"'.format(model_path)
            )
            tokenizer = dstc_utils.get_tokenizer(self.model_name)
        return tokenizer

    def _get_model(self, model):
        model_class = dstc_utils.get_model_class(self.model_name, self.is_multi_head)
        if isinstance(model, str):
            model_path = self.project_root / model
            if model_class is not GPT2MultiLMHeadModel:
                return model_class.from_pretrained(model_path).cuda()
            model_args = self.mh_fact if model_class == GPT2MultiLMHeadModel else {}
            model_kwargs = (
                {"tok": self.tokenizer, "is_inference": True}
                if model_class == GPT2MultiLMHeadModel
                else {}
            )
            return model_class.from_pretrained(
                model_path, model_args, model_kwargs
            ).cuda()
        if isinstance(model, model_class):
            if isinstance(model, GPT2MultiLMHeadModel):
                model.tok = self.tokenizer
                model.is_inference = True
            return model.cuda()
        raise ValueError(
            "model must be either a string or a model class, but model is:{model}"
        )

    def _get_datamodule(self, test_setting: str) -> BaseDataModule:
        if self.cfg.is_multi_head:
            dm = MultiLMHeadDatamodule(
                DataModuleConfig.from_inference_config(
                    self.cfg, domain_setting=test_setting
                ),
                self.cfg.mh_fact,
            )
        else:
            dm = TodDataModule(
                DataModuleConfig.from_inference_config(
                    self.cfg,
                    domain_setting=test_setting,
                )
            )
        return dm


    @classmethod
    def from_trainer_config(
        cls, trainer_config: TrainerConfig, model: GPT2LMHeadModel
    ) -> "InferenceConfig":
        return cls(
            num_workers=trainer_config.num_workers,
            data_split_percent=trainer_config.data_split_percent,
            eval_batch_size=trainer_config.eval_batch_size,
            test_batch_size=trainer_config.test_batch_size,
            max_token_len=trainer_config.max_token_len,
            raw_data_root=trainer_config.raw_data_root,
            project_root=trainer_config.project_root,
            data_prep_out_root=trainer_config.data_prep_out_root,
            num_test_dialogs=trainer_config.num_dialogs[2],
            delexicalize=trainer_config.delexicalize,
            model=model,
            model_name=trainer_config.model_name,
            generate_max_len=trainer_config.generate_max_len,
            test_domain_settings=trainer_config.test_domain_settings,
            num_turns=trainer_config.num_turns,
            overwrite=trainer_config.overwrite,
            out_dir=trainer_config.out_dir,
            tokenizer=trainer_config.tokenizer,
            test_prompt_max_len=trainer_config.test_prompt_max_len,
            is_multi_task=trainer_config.is_multi_task,
            is_multi_head=trainer_config.is_multi_head,
            mh_fact=trainer_config.mh_fact,
            is_multi_decoder=trainer_config.is_multi_decoder,
            multi_tasks=trainer_config.multi_tasks,
            should_add_schema=trainer_config.should_add_schema,
            should_add_sys_actions=trainer_config.should_add_sys_actions,
            should_add_user_actions=trainer_config.should_add_user_actions,
            context_type=trainer_config.context_type,
            should_add_service_results=trainer_config.should_add_service_results,
            postprocess_generation=trainer_config.postprocess_generation,
            datamodule=trainer_config.datamodule,
            # contrastive_model=trainer_config.contrastive_model,
        )
