from generation.multi_head_generation import MultiHeadGeneration
from generation.multi_task_generation import MultiTaskGeneration
from generation.simple_generation import SimpleGeneration


class GenerationHandlerFactory:
    @classmethod
    def get_handler(self, cfg):
        if cfg.is_multi_task:
            return MultiTaskGeneration(cfg.model, cfg.tokenizer, cfg.multi_tasks)
        if cfg.is_multi_head:
            return MultiHeadGeneration(cfg.model, cfg.tokenizer)
        return SimpleGeneration(cfg.model, cfg.tokenizer)
