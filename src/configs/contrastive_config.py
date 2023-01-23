
from pathlib import Path
from configs.trainer_config import TrainerConfig
from my_enums import ContrastiveConstants


class ContrastiveConfig:
    def __init__(
        self,
        project_root: str = "/mounts/u-amo-d0/grad/adibm/data/projects/ZSToD",
        data_prep_out_root: str = "processed_data/simple_tod",
        raw_data_root: str = "data/dstc8-schema-guided-dialogue/",
        contrastive_model_name: str = "sentence-transformers/stsb-distilroberta-base-v2",
        tokenizer_name: str = "gpt2",
        model: str = None,
        data_split_percent: list[float] = None,
        eval_batch_size: int = 6,
        test_batch_size: int = 32,
        contrastive_train_batch_size: int = 8,
        num_dialogs: list[int] = None,
        num_turns: int = 10,
        num_workers: int = 8,
        overwrite: list[bool] = None,
        train_domain_setting: str = "ALL",
        test_domain_settings: list[str] = None,
        out_dir: str = "results",
        train_epochs: int = 2,
        logging_dir: str = "logs",
        logging_steps: int = 50,
        eval_accumulation_steps: int = 5,
        is_multi_task: bool = False,
        multi_tasks: list[int] = None,
        contrast_with: list[str] = None,
        single_action_neg_samples: int = 10,
        should_add_dsts: bool = False,
        contrastive_max_token_len: int = 512,
        data_prep_multi_process: bool = True,
    ):
        self.project_root = Path(project_root)
        self.data_prep_out_root = self.project_root / data_prep_out_root
        self.raw_data_root = Path(raw_data_root)
        self.contrastive_model_name = contrastive_model_name
        self.model = self.project_root / model if model else None
        self.data_split_percent = data_split_percent or [1, 1, 1]
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size
        self.contrastive_train_batch_size = contrastive_train_batch_size
        self.num_dialogs = num_dialogs or [20, 5, 10]
        self.num_turns = num_turns
        self.num_workers = num_workers
        self.overwrite = overwrite or [False, False, False]
        self.train_domain_setting = train_domain_setting
        self.test_domain_settings = test_domain_settings or ["ALL", "SEEN", "UNSEEN"]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.train_epochs = train_epochs
        self.logging_dir = Path(logging_dir)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.logging_steps = logging_steps
        self.eval_accumulation_steps = eval_accumulation_steps
        self.is_multi_task = is_multi_task
        self.multi_tasks = multi_tasks or [1, 1, 1]
        self.contrast_with = (
            contrast_with if contrast_with else [ContrastiveConstants.USER_ACT]
        )
        self.single_action_neg_samples = single_action_neg_samples
        self.should_add_user_actions = (
            True if ContrastiveConstants.USER_ACT in self.contrast_with else False
        )
        self.should_add_dsts = should_add_dsts
        self.tokenizer_name = tokenizer_name
        self.contrastive_max_token_len = contrastive_max_token_len
        self.data_prep_multi_process = data_prep_multi_process
    
    @classmethod
    def from_trainer_config(self, trainer_cfg: TrainerConfig) -> "ContrastiveConfig":
        return self(
            project_root=trainer_cfg.project_root,
            data_prep_out_root=trainer_cfg.data_prep_out_root,
            raw_data_root=trainer_cfg.raw_data_root,
            out_dir=trainer_cfg.out_dir,
            contrastive_model_name=trainer_cfg.contrastive_model_name,
            tokenizer_name=trainer_cfg.tokenizer_name,
            num_dialogs=trainer_cfg.num_dialogs,
            num_turns=trainer_cfg.num_turns,
            num_workers=trainer_cfg.num_workers,
            overwrite=trainer_cfg.overwrite,
            train_domain_setting=trainer_cfg.train_domain_setting,
            is_multi_task=trainer_cfg.is_multi_task,
            contrastive_max_token_len=trainer_cfg.contrastive_max_token_len,
            should_add_dsts=trainer_cfg.should_add_dsts,
            contrast_with=trainer_cfg.contrast_with,
            single_action_neg_samples=trainer_cfg.single_action_neg_samples,
            train_epochs=trainer_cfg.contrastive_train_epochs,
            contrastive_train_batch_size=trainer_cfg.contrastive_train_batch_size,
            data_prep_multi_process=trainer_cfg.data_prep_multi_process,
        )
