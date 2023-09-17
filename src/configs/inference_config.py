from __future__ import annotations

from pathlib import Path
import re
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModel, AutoModelForCausalLM
from configs.task_arithmetic_config import TaskArithmeticConfig

import torch
from generation.generation_base import GenerationBase
from generation.generation_handler_factory import GenerationHandlerFactory
from vllm import LLM
from multi_head.mh_dataclasses import MultiHeadDictFactory
from multi_head.mh_datamodule import MultiLMHeadDatamodule
from multi_head.mh_model import GPT2MultiLMHeadModel
from my_enums import ContextType, MultiTaskNames, SpecialTokens, Steps
from simple_tod_dataclasses import TodTestDataBatch
from tod.turns.zs_tod_turn import TodTurnCsvRow, TodTurnMultiTaskCsvRow
from tod_datamodules import TodDataModule
import utils
import dstc.dstc_utils as dstc_utils
from typing import TYPE_CHECKING, Tuple
from configs.dm_config import DataModuleConfig

from peft import PeftModelForCausalLM, PeftConfig, PeftModel, get_peft_model

if TYPE_CHECKING:
    from configs.trainer_config import TrainerConfig
    from base_datamodule import BaseDataModule, StepData


class InferenceConfig:
    def __init__(
        self,
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 100,
        max_token_len: int = 1024,
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        project_root: str = "/mounts/u-amo-d0/grad/adibm/data/projects/ZSToD/",
        data_prep_out_root: str = "processed_data/simple_tod",
        predictions_log_dir: str = "predictions_logs",
        num_test_dialogs: int = 17,
        delexicalize: bool = False,
        model: str = "",
        model_name: str = "gpt2",
        generate_max_len: int = 1024,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_percentage: float = 1.0,
        test_domain_settings: list[str] = None,
        create_data_from_train: bool = False,
        create_data_from_train_splits: list[float] = None,
        out_dir: str = "results",
        tokenizer: AutoTokenizer = None,
        test_prompt_max_len: int = 750,
        base_model_name: str = "",
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
        datamodule: "BaseDataModule" = None,
        test_num_turns_groups: list[Tuple[int, int]] = None,
        train_step_data: "StepData" = None,
        quantization: bool = False,
        quantization_dtype: int = 8,
        num_train_dialogs: int = 1,
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
        self.quantization = quantization
        self.quantization_dtype = quantization_dtype
        self.model_name = model_name
        self.tokenizer = tokenizer if tokenizer else self._get_tokenizer(model)
        self.mh_fact = (
            mh_fact
            if mh_fact
            else MultiHeadDictFactory(self.tokenizer)
            if is_multi_head
            else None
        )

        # self.model = self.model.merge_adapter()
        # self.model = self.model.merge_and_unload()
        self.generate_max_len = generate_max_len
        self.train_domain_percentage = train_domain_percentage
        self.test_domain_settings = test_domain_settings or [
            ["all"],
            ["seen"],
            ["unseen"],
        ]
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.out_dir = out_dir
        self.test_prompt_max_len = test_prompt_max_len
        self.predictions_log_dir = Path(predictions_log_dir)
        self.predictions_log_dir.mkdir(parents=True, exist_ok=True)
        self.is_multi_task = is_multi_task
        self.base_model_name = base_model_name
        self.multi_tasks = (
            MultiTaskNames.get_multi_task_names(multi_tasks)
            if self.is_multi_task
            else None
        )
        self.model = self._get_model(model)
        self.model.eval()
        print(
            f"Inference: Model Size of {type(self.model)}: {dstc_utils.get_model_size(self.model)}"
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

        self.generation_handler: GenerationBase = GenerationHandlerFactory.get_handler(
            self
        )
        # self.contrastive_model = contrastive_model
        self.data_prep_multi_process = data_prep_multi_process
        self.wandb = wandb
        self.test_num_turns_groups = test_num_turns_groups
        self.train_step_data = train_step_data
        self.create_data_from_train = create_data_from_train
        self.create_data_from_train_splits = create_data_from_train_splits or [0.1, 0.1]
        self.num_train_dialogs = num_train_dialogs
        self.datamodule = datamodule or self._get_datamodule(self.test_domain_settings)

    def _get_tokenizer(self, model_path_str: str):
        model_path: Path = self.project_root / model_path_str
        try:
            # with specifig checkpoint number (results/train/checkpoint-1000)
            tok_path = model_path.parent.parent.parent / "tokenizer"
            # checkpoint not provided (results/train)
            if not tok_path.exists():
                tok_path = model_path.parent.parent / "tokenizer"
            if not tok_path.exists():
                tok_path = self.model_name
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except OSError:
            self.logger.info(
                'Could not find tokenizer for model "{}"'.format(model_path)
            )
            tokenizer = dstc_utils.get_tokenizer(self.model_name)
        return tokenizer

    def _get_model(self, model):
        model_class = dstc_utils.get_model_class(self.model_name, self.is_multi_head)
        if isinstance(model, str) or isinstance(model, Path):
            m_path = Path(model)
            model_path = (
                self.project_root / m_path if not m_path.is_absolute() else m_path
            )
            if model_class is not GPT2MultiLMHeadModel:
                if self.is_multi_task:
                    return self.load_multi_task_quantized_base_model(
                        self.model_name, model_path, self.multi_tasks
                    )
                if not self.quantization:
                    return model_class.from_pretrained(model_path).cuda()
                # return model_class.from_pretrained(model_path).cuda()
                device_map = "auto"
                if self.quantization_dtype == 16:
                    device_map = None
                return utils.load_quantized_model(
                    model_path, self.tokenizer, is_inference=True, device_map=device_map
                )
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
        if isinstance(model, PeftModelForCausalLM):
            return model
        raise ValueError(
            "model must be either a string or a model class, but model is:{model}"
        )

    def load_lora_adapter_model(self, model_dir):
        model_dir = Path(model_dir)
        model = utils.get_8bit_model(
            self.model_name, is_inference=True, device_map=None
        )
        model.resize_token_embeddings(len(self.tokenizer))
        model = get_peft_model(model, utils.get_lora_config(self.model_name))
        model.load_adapter(model_dir, "default")
        return model

    def load_multi_task_quantized_base_model(
        self, model_name: str, model_dir: str, tasks: list[MultiTaskNames]
    ) -> AutoModel:
        model_dir = Path(model_dir)
        model = utils.get_8bit_model(model_name, is_inference=True)
        model.resize_token_embeddings(len(self.tokenizer))
        model = get_peft_model(model, utils.get_lora_config(self.model_name))
        for task in tasks:
            adap_path = model_dir / task.value
            # model = PeftModel.from_pretrained(model, adap_path, adapter_name=task.value)
            # model.set_adapter(task.value)
            model.load_adapter(adap_path, task.value)

        return model

    def _get_datamodule(self, test_setting: str) -> BaseDataModule:
        dm_config = DataModuleConfig.from_inference_config(
            self, domain_setting=test_setting, train_step_data=self.train_step_data
        )
        if self.is_multi_head:
            return MultiLMHeadDatamodule(
                dm_config,
                self.mh_fact,
            )
        turn_cls = TodTurnMultiTaskCsvRow if self.is_multi_task else TodTurnCsvRow
        return TodDataModule(
            dm_config, steps=[Steps.TEST.value], tod_turn_row_cls=turn_cls
        )

    @classmethod
    def from_trainer_config(
        cls, trainer_config: TrainerConfig, model: str
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
            test_num_turns_groups=trainer_config.test_num_turns_groups,
            quantization=trainer_config.quantization,
            quantization_dtype=trainer_config.quantization_dtype,
            # contrastive_model=trainer_config.contrastive_model,
        )

    @classmethod
    def from_task_arithmetic_config(
        cls,
        task_arithmetic_config: TaskArithmeticConfig,
        model: GPT2LMHeadModel,
        tokenizer: AutoTokenizer,
        domain_settings: list[str],
    ) -> "InferenceConfig":
        return cls(
            model=model,
            train_step_data=task_arithmetic_config.train_step_data,
            project_root=task_arithmetic_config.project_root,
            tokenizer=tokenizer,
            test_domain_settings=domain_settings,
            create_data_from_train=task_arithmetic_config.create_data_from_train,
            create_data_from_train_splits=task_arithmetic_config.create_data_from_train_splits,
            num_test_dialogs=task_arithmetic_config.num_test_dialogs,
            test_batch_size=task_arithmetic_config.test_batch_size,
            postprocess_generation=task_arithmetic_config.postprocess_generation,
            data_split_percent=task_arithmetic_config.data_split_percent,
            quantization=task_arithmetic_config.quantization,
        )
