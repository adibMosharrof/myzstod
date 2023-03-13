import hydra
from omegaconf import DictConfig
from configs.inference_config import InferenceConfig
from configs.task_arithmetic_config import TaskArithmeticConfig
from inference import Inference
from task_vector.task_vector import TaskVector
import utils
from transformers import GPT2LMHeadModel, AutoTokenizer


class TaskArithmetic:
    def __init__(
        self,
        cfg: TaskArithmeticConfig,
    ):
        self.cfg = cfg

    def run(self):
        model_a = GPT2LMHeadModel.from_pretrained(self.cfg.model_a.path)
        model_b = GPT2LMHeadModel.from_pretrained(self.cfg.model_b.path)
        model_multi_domain = GPT2LMHeadModel.from_pretrained(
            self.cfg.model_multi_domain.path
        )

        tok_path = self.cfg.model_a.path.parent.parent / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        base_model = GPT2LMHeadModel.from_pretrained(self.cfg.model_name)
        base_model.resize_token_embeddings(len(tokenizer))

        task_vector_a = TaskVector(base_model, model_a)
        task_vector_b = TaskVector(base_model, model_b)

        task_vector_a_b = task_vector_a.__add__(task_vector_b)
        scaling_coef = 0.7
        # multi_model_using_task_vector = task_vector_a_b.apply_to(base_model)
        multi_model_using_task_vector = task_vector_a_b.apply_to(base_model, scaling_coef)
        inf_ta = Inference(
            InferenceConfig.from_task_arithmetic_config(
                self.cfg,
                multi_model_using_task_vector,
                tokenizer,
                self.cfg.model_multi_domain.domains,
            )
        )
        inf_ta.test()

        # inf_base = Inference(
        #     InferenceConfig.from_task_arithmetic_config(
        #         self.cfg, model_multi_domain, tokenizer
        #     )
        # )
        # inf_base.test()

        a = 1


@hydra.main(config_path="../config/task_arithmetic/", config_name="task_arithmetic")
def hydra_start(cfg: DictConfig) -> None:
    task_arithmetic_cfg = TaskArithmeticConfig(**cfg)
    utils.init_wandb(task_arithmetic_cfg, cfg, "task_arithmetic")
    ta = TaskArithmetic(task_arithmetic_cfg)
    ta.run()


if __name__ == "__main__":
    hydra_start()
