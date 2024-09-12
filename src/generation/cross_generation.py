from generation.generation_base import GenerationBase


class CrossGeneration(GenerationBase):
    def _get_generation(self, batch, min_len: int, max_len: int):

        gen = self.model.generate(
            inputs=batch.input_ids,
            attention_mask=batch.attention_masks,
            max_length=max_len,
            do_sample=False,
            use_cache=True,
            top_k=50,
            top_p=0.92,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            schema_tokens=batch.schema_tokens,
        )
        return self.pad_gen_to_max_len(gen, max_len)

    def remove_context(self, gen, context_len: int, max_len: int):
        return gen[:, context_len:]

    def move_to_gpu(self, batch, accelerator):
        batch_gpu = super().move_to_gpu(batch, accelerator)
        batch_gpu.schema_tokens = batch.schema_tokens.to(accelerator.device)
        return batch_gpu
