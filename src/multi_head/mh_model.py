from typing import Optional, Tuple, Union
from dotmap import DotMap
import torch
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from multi_head.mh_dataclasses import MultiHeadDict
from dataclasses import asdict


class GPT2MultiLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.lm_heads = {
            k: nn.Linear(config.n_embd, config.vocab_size, bias=False)
            for k in MultiHeadDict.head_names()
        }

        # print(self.get_memory_footprint())

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        all_head_names = MultiHeadDict.head_names()
        loss = dict.fromkeys(all_head_names, None)
        all_transformer_outputs = dict.fromkeys(all_head_names, None)
        lm_logits = dict.fromkeys(all_head_names, None)
        for head_name in all_head_names:
            transformer_outputs = self.transformer(
                input_ids[head_name],
                past_key_values=past_key_values,
                attention_mask=attention_mask[head_name],
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]
            all_transformer_outputs[head_name] = transformer_outputs
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.transformer.first_device)
                hidden_states = hidden_states.to(self.lm_heads[head_name].weight.device)

            lm_logits[head_name] = self.lm_heads[head_name](hidden_states)

            if labels[head_name] is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[head_name][..., :-1, :].contiguous()
                shift_labels = labels[head_name][..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss[head_name] = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss={loss[head_name] for head_name in all_head_names},
            logits={lm_logits[head_name] for head_name in all_head_names},
            past_key_values={
                all_transformer_outputs[head_name].past_key_values
                for head_name in all_head_names
            },
            hidden_states={
                all_transformer_outputs[head_name].hidden_states
                for head_name in all_head_names
            },
            attentions={
                all_transformer_outputs[head_name].attentions
                for head_name in all_head_names
            },
            cross_attentions={
                all_transformer_outputs[head_name].cross_attentions
                for head_name in all_head_names
            },
        )

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        for key, value in self.lm_heads.items():
            self.lm_heads[key] = value.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        for key, value in self.lm_heads.items():
            self.lm_heads[key] = value.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        # return [value for key, value in self.lm_heads.items()]
        return self.lm_heads

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            for key, oe in output_embeddings.items():
                if oe is not None:
                    self._tie_or_clone_weights(oe, self.get_input_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(
            self.config, "tie_encoder_decoder", False
        ):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder, self.base_model_prefix
            )

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if (
            self.get_output_embeddings() is not None
            and not self.config.tie_word_embeddings
        ):
            old_lm_heads = self.get_output_embeddings()
            for key, old_lm_head in old_lm_heads:
                new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
                self.set_output_embeddings(new_lm_head, key)

    def set_output_embeddings(self, new_embeddings, key):
        self.lm_heads[key] = new_embeddings
