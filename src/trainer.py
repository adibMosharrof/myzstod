from collections import defaultdict
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig
import hydra
import omegaconf
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    logging,
    EarlyStoppingCallback,
    IntervalStrategy,
    GPT2Config,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
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
from multi_head.mh_model import GPT2MultiLMHeadModel
from tod_datamodules import TodDataModule
import os
import warnings
import my_enums
import dstc.dstc_utils as dstc_utils
import utils
from sentence_transformers import SentenceTransformer
from accelerate import Accelerator

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_DEBUG"] = "INFO"
import argparse
import wandb
from my_enums import Steps, TrainingStage

from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_int8_training,
)


class SimpleTODTrainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
    ) -> None:
        self.cfg = trainer_config

    def print_cuda_info(self, step=""):
        if step:
            print(f"Step: {step}")
        print(torch.cuda.memory_allocated() / 1024**2)
        print(torch.cuda.memory_cached() / 1024**2)

    def run(self):
        self.print_cuda_info("init")
        # heads_to_prune = defaultdict(list)
        # for layer in range(12):
        #     for head in range(12):
        #         if head < self.cfg.n_head and layer < self.cfg.n_layer:
        #             continue
        #         heads_to_prune[layer].append(head)
        # model.prune_heads(heads_to_prune)
        current_dir = Path(os.getcwd())
        print(str(current_dir))
        self.cfg.datamodule = self._get_dm()
        # self.cfg.tokenizer = dstc_utils.get_trained_tokenizer(self.cfg)
        if self.cfg.train_model_path:
            pretrained_model_path = str(
                self.cfg.project_root / self.cfg.train_model_path
            )
        else:
            pretrained_model_path = self.pretrain_model(self.cfg.datamodule)
        self.print_cuda_info("after pretrain")
        self._setup_contrastive()
        self.print_cuda_info("contrastive model created")
        torch.cuda.empty_cache()
        self.print_cuda_info("empty cache before training")
        if self.cfg.two_step_training:
            out_dir = self.train_model(pretrained_model_path, self.cfg.datamodule)
            full_out_dir = str(current_dir / out_dir)
        else:
            full_out_dir = str(current_dir / pretrained_model_path)
            # full_out_dir = pretrained_model_path
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
        print(full_out_dir)

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
        trainer = Trainer(
            model=model_train,
            args=training_args,
            train_dataset=dm.datasets[my_enums.Steps.TRAIN.value],
            eval_dataset=dm.datasets[my_enums.Steps.DEV.value],
            data_collator=collator,
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
        return TrainingArguments(
            output_dir=str(self.cfg.out_dir / step_name),
            num_train_epochs=epochs,
            logging_steps=self.cfg.logging_steps,
            load_best_model_at_end=True,
            save_strategy=IntervalStrategy.STEPS,
            save_total_limit=5,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=self.cfg.eval_steps,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            metric_for_best_model="eval_loss",
            eval_accumulation_steps=self.cfg.eval_accumulation_steps,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=self.cfg.logging_dir,
            dataloader_num_workers=self.cfg.num_workers,
            report_to="wandb",
            fp16=self.cfg.fp16,
            dataloader_drop_last=True,
            run_name=step_name,
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
        return model

    def get_quantized_model(self, path: Path = None):
        if path:
            model = utils.load_quantized_model(path, self.cfg.tokenizer)
        else:
            current_device = Accelerator().process_index
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                load_in_8bit=True,
                device_map="auto",
            )
            model.resize_token_embeddings(len(self.cfg.tokenizer))
        if "gpt-neox" in self.cfg.model_name:
            model = prepare_model_for_int8_training(
                model,
                output_embedding_layer_name="embed_out",
                layer_norm_names=["layer_norm", "layernorm"],
            )
        else:
            model = prepare_model_for_int8_training(model)
        target_modules = None
        # target_modules = ["q_proj", "v_proj"]
        if "gpt-neox" in self.cfg.model_name:
            target_modules = [
                "query_key_value",
                "xxx",
            ]  # workaround to use 8bit training on this model

        if self.cfg.save_wte:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                base_model_name_or_path=self.cfg.model_name,
                modules_to_save=["wte"],
            )
        else:
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                base_model_name_or_path=self.cfg.model_name,
            )            

        model = get_peft_model(model, config)
        if model.active_peft_config.base_model_name_or_path is None:
            model.active_peft_config.base_model_name_or_path = self.cfg.model_name
        self.print_trainable_parameters(model)
        model.gradient_checkpointing_enable()
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
        # model.lm_head = CastOutputToFloat(model.lm_head)
        # model.resize_token_embeddings(len(self.cfg.tokenizer))
        print(f"Model Size of {type(model)}: {dstc_utils.get_model_size(model)}")
        # pre_trainer = Trainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=dm.datasets[Steps.TRAIN.value],
        #     eval_dataset=dm.datasets[Steps.DEV.value],
        #     data_collator=dm.pretraining_collator,
        #     callbacks=[
        #         EarlyStoppingCallback(
        #             early_stopping_patience=self.cfg.early_stopping_patience
        #         ),
        #     ],
        # )
        pre_trainer = self._get_trainer(
            model, dm, training_args, training_stage=TrainingStage.PRETRAIN
        )
        # with torch.cuda.amp.autocast():
        #     pre_trainer.train()
        model.config.use_cache = False
        model.train()
        pre_trainer.train()
        pre_trainer.save_model()
        model.save_pretrained(training_args.output_dir)
        return training_args.output_dir
        return model

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

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


@hydra.main(config_path="../config/trainer/", config_name="arithmetic_trainer")
def hydra_start(cfg: DictConfig) -> None:
    trainer_cfg = TrainerConfig(**cfg)
    # utils.init_wandb(trainer_cfg, cfg, "training")
    stt = SimpleTODTrainer(trainer_cfg)
    stt.run()


if __name__ == "__main__":
    hydra_start()
