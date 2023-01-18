from typing import Callable, Iterable, List, Optional, Tuple, Union
from dotmap import DotMap
import torch
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.utils import GenerateOutput
from transformers.generation.beam_constraints import Constraint
from multi_head.mh_dataclasses import MultiHeadDictFactory
from dataclasses import asdict
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GPT2MultiLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, mh_fact: MultiHeadDictFactory, kwargs=None):
        super().__init__(config)
        self.mh_fact = mh_fact
        self.tok = kwargs.get("tok", None)
        self.is_inference = kwargs.get("is_inference", False)
        self.head_names = self.mh_fact.get_head_names()
        self.lm_heads = nn.ModuleDict(
            {
                k: nn.Linear(config.n_embd, config.vocab_size, bias=False)
                for k in self.head_names
            }
        )
        self.lm_head = None

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        penalty_alpha: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[int, float]] = None,
        suppress_tokens: Optional[List[int]] = None,
        begin_suppress_tokens: Optional[List[int]] = None,
        forced_decoder_ids: Optional[List[List[int]]] = None,
        **model_kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        all_out = []
        for head_instance in self.mh_fact.get_head_instances():
            out = super().generate(
                inputs[head_instance.name],
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=head_instance.eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                max_length=max_length,
                head_name=head_instance.name,
                # head_eos_token_id=head_instance.eos_token_id,
            )
            if out.shape[1] < max_length:
                out = torch.cat(
                    [
                        out,
                        torch.full(
                            (out.shape[0], max_length - out.shape[1]),
                            pad_token_id,
                            dtype=out.dtype,
                            device=out.device,
                        ),
                    ],
                    dim=1,
                )
            all_out.append(out)
        return all_out
        # return torch.cat(all_out, dim=0)
    
    def forward_single_head(
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
        lm_head: Optional[nn.Module] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
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

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(lm_head.weight.device)

        lm_logits = lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    def forward(
        self,
        input_ids: Optional[dict[str, torch.LongTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[dict[str, torch.LongTensor]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[dict[str, torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        head_name: Optional[str] = None,
        head_eos_token_id: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        if self.is_inference:
            return self.forward_single_head(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                lm_head=self.lm_heads[head_name],
            )

        outs: list[CausalLMOutputWithCrossAttentions] = []
        for head_name in self.head_names:
            inputs = input_ids[head_name]
            mask = attention_mask[head_name]
            out = self.forward_single_head(
                input_ids=inputs,
                past_key_values=past_key_values,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels[head_name],
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                lm_head=self.lm_heads[head_name],
            )
            outs.append(out)

        return CausalLMOutputWithCrossAttentions(
            loss=torch.sum(torch.stack([out.loss for out in outs])),
            logits=torch.stack([out.logits for out in outs]),
            past_key_values=[out.past_key_values for out in outs],
            hidden_states=[out.hidden_states for out in outs],
            attentions=[out.attentions for out in outs],
            cross_attentions=[out.cross_attentions for out in outs],
        )

    def forward_old(
        self,
        input_ids: Optional[dict[str, torch.LongTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[dict[str, torch.LongTensor]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[dict[str, torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        all_loss = dict.fromkeys(self.head_names, None)
        all_transformer_outputs = dict.fromkeys(self.head_names, None)
        lm_logits = dict.fromkeys(self.head_names, None)
        for head_name in self.head_names:
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
                all_loss[head_name] = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((all_loss,) + output) if all_loss is not None else output
        total_loss = sum(all_loss.values())
        return CausalLMOutputWithCrossAttentions(
            loss=total_loss,
            logits={lm_logits[head_name] for head_name in self.head_names},
            past_key_values={
                all_transformer_outputs[head_name].past_key_values
                for head_name in self.head_names
            },
            hidden_states={
                all_transformer_outputs[head_name].hidden_states
                for head_name in self.head_names
            },
            attentions={
                all_transformer_outputs[head_name].attentions
                for head_name in self.head_names
            },
            cross_attentions={
                all_transformer_outputs[head_name].cross_attentions
                for head_name in self.head_names
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
        # self.lm_head = self.lm_head.to(self.transformer.first_device)
        for key, value in self.lm_heads.items():
            # self.lm_heads[key] = value.to(self.transformer.first_device)
            self.lm_heads[key] = self.lm_heads[key].to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        # self.lm_head = self.lm_head.to("cpu")
        for key, value in self.lm_heads.items():
            # self.lm_heads[key] = value.to("cpu")
            self.lm_heads[key] = self.lm_heads[key].to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
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

    def estimate_tokens(self, input_dict: dict[dict[str, torch.Tensor]]) -> int:
        if not hasattr(self, "warnings_issued"):
            self.warnings_issued = {}

        if self.main_input_name in input_dict:
            elements = [
                input_dict[self.main_input_name][head_name].numel()
                for head_name in self.head_names
            ]
            return sum(elements)
        elif "estimate_tokens" not in self.warnings_issued:
            logger.warning(
                "Could not estimate the number of tokens of the input, floating-point operations will not be computed"
            )
            self.warnings_issued["estimate_tokens"] = True
        return 0

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        inputs = super().prepare_inputs_for_generation(input_ids, past=past, **kwargs)
        inputs["head_name"] = kwargs.get("head_name", kwargs)
        inputs["head_eos_token_id"] = kwargs.get("head_eos_token_id", kwargs)
        return inputs
