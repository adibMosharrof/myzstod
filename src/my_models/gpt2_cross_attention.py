import torch
from torch import nn
from transformers import GPT2Model, GPT2PreTrainedModel, GPT2Config


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, hidden_states, key_value_states, attention_mask=None):
        # Create query from input (dialog history), and key/value from schema
        query = self.query(hidden_states)
        key = self.key(key_value_states)
        value = self.value(key_value_states)

        # Attention mechanism
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / (
            key.size(-1) ** 0.5
        )

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute the attention output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self.resid_dropout(self.proj(attn_output))
        return attn_output


class GPT2WithCrossAttention(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gpt2 = GPT2Model(config)
        self.cross_attention = CrossAttention(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self, input_ids, schema_ids, attention_mask=None, schema_attention_mask=None
    ):
        # Forward pass for dialog history using GPT-2
        transformer_outputs = self.gpt2(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden_states = transformer_outputs.last_hidden_state

        # Forward pass for schema using GPT-2 (reuse GPT-2 for schema processing)
        schema_outputs = self.gpt2(
            input_ids=schema_ids, attention_mask=schema_attention_mask
        )
        schema_hidden_states = schema_outputs.last_hidden_state

        # Apply cross-attention
        cross_attn_output = self.cross_attention(hidden_states, schema_hidden_states)

        # Combine self-attention and cross-attention (via residual connection here)
        final_hidden_states = hidden_states + cross_attn_output

        # Pass the final hidden states to the LM head for generation
        lm_logits = self.lm_head(final_hidden_states)

        return lm_logits

    @staticmethod
    def from_pretrained(self, model_name_or_path, *args, **kwargs):
        config = GPT2Config.from_pretrained(model_name_or_path)
        model = self(config)
        model.gpt2 = GPT2Model.from_pretrained(model_name_or_path)
        return model
