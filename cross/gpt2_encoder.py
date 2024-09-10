from prompts.nlg_prompt_manager import CrossAttentionPrompt

from schema_encoder import SchemaEncoder
from transformers import AutoModel, AutoTokenizer
import torch


class Gpt2Encoder(SchemaEncoder):
    def __init__(self, cfg, model: AutoModel, tokenizer: AutoTokenizer):
        super().init(cfg, model, tokenizer)

    def get_embedding(self, prompt: str):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def encode(self, domain: str, schema: str) -> str:
        prompt = CrossAttentionPrompt.get_schema_prompt(domain, schema)
        prompt_embedding = self.get_embedding(prompt)
        return prompt_embedding
