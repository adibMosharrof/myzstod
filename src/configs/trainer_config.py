from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from base_datamodule import BaseDataModule
import dstc.dstc_utils as dstc_utils
from pathlib import Path
from multi_head.mh_dataclasses import MultiHeadDictFactory
from my_enums import ContextType


class TrainerConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/data/projects/ZSToD/",
        data_prep_out_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        model_name: str = "gpt2",
        contrastive_model_name: str = "sentence-transformers/stsb-roberta-base-v2",
        tokenizer_name: str = None,
        num_workers: int = 8,
        data_split_percent: list[float] = None,
        early_stopping_patience: int = 3,
        eval_steps: int = 500,
        eval_batch_size: int = 6,
        test_batch_size: int = 32,
        train_batch_size: int = 8,
        pretrain_batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        n_layer: int = 12,
        n_head: int = 12,
        contrastive_train_batch_size: int = 100,
        num_dialogs: list[int] = None,
        delexicalize: bool = False,
        num_turns: int = 10,
        overwrite: list[bool] = None,
        train_domain_percentage: float = 1.0,
        train_domain_settings: list[str] = None,
        dev_domain_settings: list[str] = None,
        test_domain_settings: list[list[str]] = None,
        create_data_from_train: bool = False,
        create_data_from_train_splits: list[float] = None,
        out_dir: str = "results",
        pretrain_epochs: int = 1,
        pretrain_model_path: str = None,
        train_model_path: str = None,
        train_epochs: int = 1,
        contrastive_train_epochs: int = 3,
        logging_dir: str = "logs",
        generate_max_len: int = 1024,
        should_test: bool = False,
        logging_steps: int = 50,
        test_prompt_max_len: int = 799,
        max_token_len: int = 1024,
        eval_accumulation_steps: int = 16,
        is_multi_head: bool = False,
        is_multi_task: bool = False,
        is_multi_decoder: bool = False,
        multi_tasks: list[int] = None,
        should_add_schema: bool = False,
        should_add_user_actions: bool = False,
        should_add_sys_actions: bool = False,
        ce_loss_weight: float = 0.50,
        contrastive_loss_weight: float = 0.50,
        contrastive_model: str = None,
        contrast_with: list[str] = None,
        contrastive_max_token_len: int = 250,
        context_type: str = ContextType.SHORT_REPR,
        should_add_service_results: bool = False,
        should_add_dsts: bool = False,
        single_action_neg_samples: int = 10,
        local_rank: int = 0,
        fp16: int = False,
        postprocess_generation: bool = False,
        wandb: any = None,
        data_prep_multi_process: bool = True,
        datamodule: "BaseDataModule" = None,
        test_num_turns_groups: list[Tuple[int, int]] = None,
        two_step_training: bool = True,
        quantization: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.data_prep_out_root = Path(data_prep_out_root)
        self.model_name = model_name
        self.num_workers = num_workers
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.early_stopping_patience = early_stopping_patience
        self.eval_steps = eval_steps
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_token_len = max_token_len
        self.num_dialogs = num_dialogs or [20, 10, 17]
        self.delexicalize = delexicalize
        self.num_turns = num_turns
        self.overwrite = overwrite or [False, False, False]
        self.out_dir = Path(out_dir)
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.contrastive_train_epochs = contrastive_train_epochs
        self.quantization = quantization
        self.dev_domain_settings = dev_domain_settings or ["seen"]
        self.train_domain_settings = train_domain_settings or ["seen"]
        self.test_domain_settings = test_domain_settings or [
            ["all"],
            ["seen"],
            ["unseen"],
        ]
        self.create_data_from_train = create_data_from_train
        self.create_data_from_train_splits = create_data_from_train_splits or [0.1, 0.1]
        self.train_domain_percentage = train_domain_percentage
        self.pretrain_model_path = pretrain_model_path
        self.train_model_path = train_model_path
        self.logging_dir = Path(logging_dir)
        self.generate_max_len = generate_max_len
        self.should_test = should_test
        self.delexicalize = delexicalize
        self.logging_steps = logging_steps
        self.train_batch_size = train_batch_size
        self.contrastive_train_batch_size = contrastive_train_batch_size
        self.pretrain_batch_size = pretrain_batch_size
        self.raw_data_root = self.project_root / raw_data_root
        self.test_prompt_max_len = test_prompt_max_len
        self.eval_accumulation_steps = eval_accumulation_steps
        self.fp16 = fp16
        self.is_multi_head = is_multi_head
        self.is_multi_task = is_multi_task
        self.is_multi_decoder = is_multi_decoder
        self.multi_tasks = (
            multi_tasks if self.is_multi_task and multi_tasks else [1, 1, 1]
        )
        self.tokenizer_name = tokenizer_name or model_name
        self.tokenizer = dstc_utils.get_tokenizer(self.tokenizer_name)
        self.should_add_schema = should_add_schema
        self.should_add_sys_actions = should_add_sys_actions
        self.should_add_user_actions = should_add_user_actions
        if test_prompt_max_len > max_token_len:
            raise ValueError("context_max_len must be less than max_token_len")
        self.ce_loss_weight = ce_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_model = contrastive_model
        self.contrast_with = contrast_with or []
        self.contrastive_max_token_len = contrastive_max_token_len
        self.context_type = context_type
        self.should_add_service_results = should_add_service_results
        
        self.contrastive_model_name = contrastive_model_name
        self.should_add_dsts = should_add_dsts
        self.single_action_neg_samples = single_action_neg_samples
        self.local_rank = local_rank
        self.postprocess_generation = postprocess_generation
        self.mh_fact = (
            MultiHeadDictFactory(self.tokenizer) if self.is_multi_head else None
        )
        self.data_prep_multi_process = data_prep_multi_process
        self.wandb = wandb
        self.test_num_turns_groups = test_num_turns_groups
        self.datamodule = datamodule
        self.two_step_training = two_step_training
