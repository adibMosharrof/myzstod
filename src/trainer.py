import sys
from pathlib import Path
from typing import Optional
import hydra
from omegaconf import DictConfig
import omegaconf
import torch
import gc
from datetime import datetime
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    IntervalStrategy,
)
from base_datamodule import BaseDataModule
from configs.contrastive_config import ContrastiveConfig
from configs.dm_config import DataModuleConfig
from configs.inference_config import InferenceConfig
from configs.trainer_config import TrainerConfig
from contrastive.contrastive import Contrastive
from contrastive.contrastive_datamodule import ContrastiveDataModule
from contrastive.contrastive_trainer import (
    ContrastiveTrainerHelper,
    ContrastiveTrainer,
)

from inference import Inference
from multi_head.mh_dataclasses import MultiHeadDictFactory
from multi_head.mh_datamodule import MultiLMHeadDatamodule
from tod.turns.turn_csv_row_factory import TurnCsvRowFactory
from tod.turns.zs_tod_turn import TodTurnMultiTaskCsvRow, TodTurnScaleGradCsvRow
from tod_datamodules import TodDataModule
import os
import warnings
import my_enums
import dstc.dstc_utils as dstc_utils
import utils
from sentence_transformers import SentenceTransformer
from deepspeed.runtime.utils import see_memory_usage

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_DEBUG"] = "INFO"
import wandb
from my_enums import Steps, TrainingStage
from transformers.trainer_pt_utils import get_parameter_names
from peft import (
    PeftModelForCausalLM,
    prepare_model_for_kbit_training,
)
import bitsandbytes as bnb
from torch import nn
import deepspeed


class SimpleTODTrainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
    ) -> None:
        self.cfg = trainer_config

    def print_cuda_info(self, step=""):
        see_memory_usage(step, force=True)

    def multi_task_run(self):
        dms = self._get_multi_task_dms()
        model_paths = []
        for dm in dms:
            model_path = self.train_multi_task_model(dm)
            model_paths.append(model_path)
            # model = self.train_multi_task_model(dm)
            torch.cuda.empty_cache()
        self.cfg.model_paths = model_paths
        if self.cfg.should_test:
            curr_dir = Path(os.getcwd())
            inf = Inference(
                InferenceConfig.from_trainer_config(
                    self.cfg, str(curr_dir / "results" / "multi_task")
                ),
                # InferenceConfig.from_trainer_config(self.cfg, model),
            )
            inf.test()
        print(str(self.cfg.out_dir.absolute()))

    def run(self):
        self.print_cuda_info("init")
        current_dir = Path(os.getcwd())
        print(str(current_dir))
        if self.cfg.is_multi_task:
            self.multi_task_run()
            return
        self.cfg.datamodule = self._get_dm()
        # self.cfg.tokenizer = dstc_utils.get_trained_tokenizer(self.cfg)
        if self.cfg.train_model_path:
            pretrained_model_path = str(
                self.cfg.project_root / self.cfg.train_model_path
            )
        else:
            pretrained_model_path = self.pretrain_model(self.cfg.datamodule)
        self.print_cuda_info("after pretrain")
        gc.collect()
        self._setup_contrastive()
        self.print_cuda_info("contrastive model created")

        torch.cuda.empty_cache()
        self.print_cuda_info("empty cache before training")
        if self.cfg.two_step_training:
            out_dir = self.train_model(pretrained_model_path, self.cfg.datamodule)
            full_out_dir = str(current_dir / out_dir)
        else:
            # full_out_dir = str(current_dir / pretrained_model_path)
            full_out_dir = pretrained_model_path
        self.print_cuda_info("after train")
        print("Training done")
        print("-" * 80)
        torch.cuda.empty_cache()
        self.print_cuda_info("empty cache before testing")
        if self.cfg.should_test:
            inf = Inference(
                InferenceConfig.from_trainer_config(self.cfg, full_out_dir),
            )
            inf.test()
        print(current_dir)

    def _get_multi_task_dms(self) -> list[BaseDataModule]:
        steps = Steps.list() if self.cfg.should_test else Steps.list()[:-1]
        return [
            TodDataModule(
                DataModuleConfig.from_trainer_config(self.cfg),
                steps=steps,
                tod_turn_row_cls=TodTurnMultiTaskCsvRow,
                task_name=task_name,
            )
            for task_name in self.cfg.multi_tasks
        ]

    def _get_dm(self) -> BaseDataModule:
        steps = Steps.list() if self.cfg.should_test else Steps.list()[:-1]
        if self.cfg.is_multi_head:
            return MultiLMHeadDatamodule(
                DataModuleConfig.from_trainer_config(self.cfg),
                steps,
                MultiHeadDictFactory(self.cfg.tokenizer),
            )
        if self.cfg.contrast_with:
            return ContrastiveDataModule(
                DataModuleConfig.from_trainer_config(self.cfg), steps
            )
        if self.cfg.is_scale_grad:
            return TodDataModule(
                DataModuleConfig.from_trainer_config(self.cfg),
                steps,
                tod_turn_row_cls=TodTurnScaleGradCsvRow,
            )
        return TodDataModule(DataModuleConfig.from_trainer_config(self.cfg), steps)

    def _setup_contrastive(self) -> Optional[AutoTokenizer]:
        if not self.cfg.contrast_with:
            return None
        if self.cfg.contrastive_model:
            model_or_path = SentenceTransformer(
                self.cfg.project_root / self.cfg.contrastive_model
            ).cuda()
        else:
            c = Contrastive(ContrastiveConfig.from_trainer_config(self.cfg))
            model_or_path = c.run()
        self.contrastive_helper = ContrastiveTrainerHelper(
            model_or_path,
            self.cfg.tokenizer,
            self.cfg.contrastive_max_token_len,
            Contrastive.get_start_end_tokens(self.cfg.contrast_with),
            self.cfg.is_multi_task,
            self.cfg.ce_loss_weight,
            self.cfg.contrastive_loss_weight,
        )
        return self.contrastive_helper.contrastive_model.tokenizer

    def _get_8bit_optimizer(self, model, training_args):
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
        }
        optimizer_kwargs["lr"] = training_args.learning_rate
        adam_bnb_optim = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            lr=training_args.learning_rate,
        )
        return (adam_bnb_optim, None)

    def _get_trainer(
        self,
        model_train: AutoModel,
        dm: TodDataModule,
        training_args: TrainingArguments,
        training_stage: TrainingStage = TrainingStage.TRAIN,
    ) -> Trainer:
        collator = (
            dm.training_collator
            if training_stage == TrainingStage.TRAIN
            else dm.pretraining_collator
        )
        if self.cfg.contrast_with:
            trainer = ContrastiveTrainer(
                model=model_train,
                args=training_args,
                train_dataset=dm.datasets[Steps.TRAIN.value],
                eval_dataset=dm.datasets[Steps.DEV.value],
                data_collator=collator,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=self.cfg.early_stopping_patience
                    )
                ],
            )
            trainer.contrastive_helper = self.contrastive_helper
            return trainer
        # optimizer = None
        # if self.cfg.quantization:
        #     optimizer = self._get_8bit_optimizer(model_train, training_args)
        # optimizer = self._get_8bit_optimizer(model_train, training_args)

        trainer = Trainer(
            model=model_train,
            args=training_args,
            train_dataset=dm.datasets[my_enums.Steps.TRAIN.value],
            eval_dataset=dm.datasets[my_enums.Steps.DEV.value],
            data_collator=collator,
            # optimizers=optimizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience
                ),
                # utils.PeftSavingCallback,
            ],
        )
        return trainer

    def _get_training_args(
        self, step_name: str, epochs: int, train_batch_size: int
    ) -> TrainingArguments:
        deepspeed_path = None
        if self.cfg.quantization_dtype == 16:
            # deepspeed_path = self.cfg.project_root / "config/ds_config.json"
            # deepspeed_path = self.cfg.project_root / "config/ds_zero2.json"
            # deepspeed_path = self.cfg.project_root / "config/ds_zero1.json"
            deepspeed_path = ""

        return TrainingArguments(
            output_dir=str(self.cfg.out_dir / step_name),
            num_train_epochs=epochs,
            logging_steps=self.cfg.logging_steps,
            load_best_model_at_end=True,
            save_strategy=IntervalStrategy.STEPS,
            save_total_limit=5,
            save_steps=self.cfg.save_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.cfg.eval_steps,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            metric_for_best_model="eval_loss",
            eval_accumulation_steps=self.cfg.eval_accumulation_steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir=self.cfg.logging_dir,
            dataloader_num_workers=self.cfg.num_workers,
            report_to="wandb",
            fp16_full_eval=self.cfg.fp16,
            fp16=self.cfg.fp16,
            dataloader_drop_last=True,
            run_name=step_name,
            learning_rate=3e-4,
            optim=self.cfg.optim,
            # sharded_ddp="zero_dp_3",
            # deepspeed=deepspeed_path,
            # fsdp_config=str(self.cfg.project_root / "config/ds_config.json"),
            # fsdp="full_shard",
        )

    def get_model_instance(self, path: str = None) -> AutoModel:
        model_class = dstc_utils.get_model_class(
            self.cfg.model_name, self.cfg.is_multi_head
        )
        if self.cfg.is_multi_head:
            model = model_class.from_pretrained(
                path or self.cfg.model_name,
                self.cfg.mh_fact,
                {"tok": self.cfg.tokenizer},
            )
        else:
            model = model_class.from_pretrained(path or self.cfg.model_name)
        model.resize_token_embeddings(len(self.cfg.tokenizer))
        return model

    def get_quantized_model(self, path: Path = None, adapter_name: str = "default"):
        if path:
            model = utils.load_quantized_model(path, self.cfg.tokenizer)
        else:
            device_map = {"": torch.cuda.current_device()}
            if self.cfg.is_scale_grad:
                model = utils.get_scalegrad_model(
                    self.cfg.model_name, self.cfg.scale_grad_gamma, device_map=None
                )
            elif self.cfg.quantization_dtype == 16:
                dtype = torch.bfloat16
                if not self.cfg.fp16:
                    dtype = torch.float32
                model = utils.get_8bit_model(
                    self.cfg.model_name, is_inference=True, device_map=None, dtype=dtype
                )
            elif self.cfg.quantization_dtype == 8:
                model = utils.get_8bit_model(self.cfg.model_name, device_map=device_map)
            elif self.cfg.quantization_dtype == 4:
                model = utils.get_4bit_model(self.cfg.model_name, device_map=device_map)
            else:
                raise ValueError(
                    f"Quantization dtype must be one of 4, 8, 16, you provided {self.cfg.quantization_dtype}"
                )
            model.resize_token_embeddings(len(self.cfg.tokenizer))

        model = prepare_model_for_kbit_training(model)

        config = utils.get_lora_config(self.cfg.model_name)

        model = PeftModelForCausalLM(model, config, adapter_name=adapter_name)
        if model.active_peft_config.base_model_name_or_path is None:
            model.active_peft_config.base_model_name_or_path = self.cfg.model_name
        model.print_trainable_parameters()
        return model

    def pretrain_model(self, dm: TodDataModule) -> str:
        if self.cfg.pretrain_model_path:
            path = self.cfg.project_root / self.cfg.pretrain_model_path
            if path.exists():
                return str(path)
        training_args = self._get_training_args(
            "pretrain", self.cfg.pretrain_epochs, self.cfg.pretrain_batch_size
        )
        model = (
            self.get_model_instance()
            if not self.cfg.quantization
            else self.get_quantized_model()
        )
        print(f"Model Size of {type(model)}: {utils.get_model_size(model)}")

        pre_trainer = self._get_trainer(
            model, dm, training_args, training_stage=TrainingStage.PRETRAIN
        )
        model.config.use_cache = False
        model.train()
        # with torch.autocast("cuda"):
        pre_trainer.train()
        if self.cfg.accelerator.is_main_process:
            pre_trainer.save_model()
            model.save_pretrained(training_args.output_dir)
        self.cfg.accelerator.wait_for_everyone()
        # return model
        # del model
        # torch.cuda.empty_cache()
        return str(Path(os.getcwd()) / training_args.output_dir)

    def train_model(self, path, dm) -> str:
        model = (
            self.get_model_instance(path)
            if not self.cfg.quantization
            else self.get_quantized_model(path)
        )
        training_args = self._get_training_args(
            "train", self.cfg.train_epochs, self.cfg.train_batch_size
        )
        trainer = self._get_trainer(
            model, dm, training_args, training_stage=TrainingStage.TRAIN
        )
        model.train()
        trainer.train()
        trainer.save_model()
        model.save_pretrained(training_args.output_dir)
        out_dir = os.getcwd()
        print("training output_dir: ", out_dir)
        return training_args.output_dir

    def train_multi_task_model(self, dm: BaseDataModule) -> AutoModel:
        training_args = self._get_training_args(
            "multi_task", self.cfg.pretrain_epochs, self.cfg.pretrain_batch_size
        )
        model = (
            self.get_model_instance()
            if not self.cfg.quantization
            else self.get_quantized_model(adapter_name=dm.task_name.value)
        )
        print(f"Model Size of {type(model)}: {utils.get_model_size(model)}")

        pre_trainer = self._get_trainer(
            model, dm, training_args, training_stage=TrainingStage.PRETRAIN
        )
        model.config.use_cache = False
        model.train()
        pre_trainer.train()
        # pre_trainer.save_model()
        model.save_pretrained(training_args.output_dir)
        # del model
        # gc.collect()
        # torch.cuda.empty_cache()
        return training_args.output_dir
        # return model


# @hydra.main(config_path="../config/trainer/", config_name="multi_adapter")
@hydra.main(config_path="../config/trainer/", config_name="simple_tod_trainer")
def hydra_start(cfg: DictConfig) -> None:
    trainer_cfg = TrainerConfig(**cfg)
    utils.init_wandb(trainer_cfg, cfg, "training")
    stt = SimpleTODTrainer(trainer_cfg)
    print(os.getcwd())
    stt.run()
    sys.stdout.close()


def create_out_dir():
    time_now = datetime.now()
    out_path_name = time_now.strftime("%Y-%m-%d/%H-%M-%S")
    out_path = "outputs" / Path(out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)
    os.chdir(out_path)


if __name__ == "__main__":
    # deepspeed.init_distributed()
    hydra_start()
    wandb.finish()
