import gc
import hydra
from omegaconf import DictConfig
from configs.inference_config import InferenceConfig
from configs.task_arithmetic_config import TaskArithmeticConfig
from inference import Inference
from task_vector.task_vector import TaskVector
import utils
import dstc.dstc_utils as dstc_utils
from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModel, AutoModelForCausalLM
from pathlib import Path
import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_int8_training,
    PeftModelForCausalLM,
)
from peft.utils import set_peft_model_state_dict


class TaskArithmetic:
    def __init__(
        self,
        cfg: TaskArithmeticConfig,
    ):
        self.cfg = cfg

    def run(self):
        # model_a = self.load_model(self.cfg.model_name)
        base_model = utils.get_8bit_model(self.cfg.model_name, is_inference=True)
        base_model.resize_token_embeddings(len(self.cfg.tokenizer))
        # model_a = self.load_model(self.cfg.model_a.path, self.cfg.tokenizer_name)
        # model_b = self.load_model(self.cfg.model_b.path, self.cfg.tokenizer_name)
        # base_model.load_adapter(self.cfg.model_a.path, "default")
        task_vector_a = self.get_task_vector(base_model, self.cfg.model_a.path)
        # task_vector_b = self.get_task_vector(base_model, self.cfg.model_b.path)

        # task_vector_a_b = task_vector_a.__add__(task_vector_b)
        scaling_coef = 1.0

        multi_model_using_task_vector = task_vector_a.apply_to(base_model, scaling_coef)
        # multi_model_using_task_vector = task_vector_a_b.apply_to(
        # multi_model_using_task_vector = task_vector_a.apply_to(base_model, scaling_coef)

        # model_multi_domain = self.load_model(self.cfg.model_multi_domain.path)
        # tok_path = self.cfg.model_multi_domain.path.parent.parent / "tokenizer"

        inf_ta = Inference(
            InferenceConfig.from_task_arithmetic_config(
                self.cfg,
                multi_model_using_task_vector,
                # model_a,
                self.cfg.tokenizer,
                # self.cfg.model_multi_domain.domains,
                self.cfg.model_a.domains,
            )
        )
        # del base_model
        # del task_vector_a
        gc.collect()
        torch.cuda.empty_cache()
        inf_ta.test()

    def get_task_vector(self, base_model, model_path):
        model = self.load_model(model_path, self.cfg.tokenizer)
        return TaskVector(base_model, model)

    def load_model(self, model_name_or_path, tokenizer_name: str):
        if not self.cfg.quantization:
            return AutoModel.from_pretrained(model_name_or_path)
        return utils.load_quantized_model(
            model_name_or_path, tokenizer_name, self.cfg.quantization_dtype
        )


@hydra.main(config_path="../config/task_arithmetic/", config_name="task_arithmetic")
def hydra_start(cfg: DictConfig) -> None:
    task_arithmetic_cfg = TaskArithmeticConfig(**cfg)
    utils.init_wandb(task_arithmetic_cfg, cfg, "task_arithmetic")
    ta = TaskArithmetic(task_arithmetic_cfg)
    ta.run()


if __name__ == "__main__":
    hydra_start()
