from generation.cross_generation import CrossGeneration
from generation.interpret_generation import InterpretGeneration
from generation.multi_head_generation import MultiHeadGeneration
from generation.multi_task_generation import MultiTaskGeneration
from generation.simple_generation import SimpleGeneration
from generation.t5_generation import T5Generation
from my_enums import ContextType
import utils


class GenerationHandlerFactory:
    @classmethod
    def get_handler(self, cfg, model=None, tokenizer=None):
        model = model or cfg.model
        tokenizer = tokenizer or cfg.tokenizer
        if cfg.is_multi_task:
            return MultiTaskGeneration(
                cfg.model,
                cfg.tokenizer,
                cfg.multi_tasks,
                cfg.model_paths,
                cfg.project_root,
            )
        if cfg.is_multi_head:
            return MultiHeadGeneration(cfg.model, cfg.tokenizer)
        if cfg.model_type.context_type == ContextType.GPT_CROSS.value:
            return CrossGeneration(model, tokenizer)
        if cfg.get("is_interpret", None):
            return InterpretGeneration(model, tokenizer)
        if utils.is_t5_model(cfg.model_type.model_name):
            return T5Generation(model, tokenizer)
        return SimpleGeneration(model, tokenizer)
