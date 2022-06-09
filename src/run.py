# import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

from my_datamodules import SimpleTodDataModule
from simple_tod_dataclasses import SpecialTokens
from trainer import MyTrainer


def run():

    model_name = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    # special_tokens = torch.tensor(SpecialTokens.list(), device=torch.device("cuda"))
    special_tokens = SpecialTokens.list()
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model = model.cuda()

    data_root = "/localdisk0/adibm/data/dstc8-schema-guided-dialogue/processed_out/"
    dm = SimpleTodDataModule(tokenizer=tokenizer, data_root=data_root)
    dm.setup()
    # train(model, dm)
    return test(dm, tokenizer)


def train(model, dm):
    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=2,
        logging_steps=10,
        load_best_model_at_end=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=38,
        per_device_eval_batch_size=38,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="logs",
    )

    # start training
    Trainer(
        model=model,
        args=training_args,
        train_dataset=dm.datasets["train"],
        eval_dataset=dm.datasets["dev"],
        data_collator=dm.my_collate,
        # data_collator=lambda data: {
        #     "input_ids": torch.stack([f[0]["input_ids"] for f in data]),
        #     "attention_mask": torch.stack([f[0]["attention_mask"] for f in data]),
        #     "labels": torch.stack([f[1]["input_ids"] for f in data]),
        # },
    ).train()
    # trainer = MyTrainer(model, training_args, dm.my_collate, dm)
    # trainer.train()


def test(dm, tokenizer):
    model = GPT2LMHeadModel.from_pretrained("results/checkpoint-89")

    # for contexts_batch, targets_batch in tqdm(dm.test_dataloader()):
    for row in tqdm(dm.test_dataloader()):
        # (_,c_id), (_,c_am), (_,t_id), (_,t_am) = row.items()
        (_, c_id), (_, c_am), (_, l_id) = row.items()
        # for a in zip(c_id, c_am, t_id, t_am):
        for a in zip(c_id, c_am, l_id):
            t_text = tokenizer.decode(a[2], skip_special_tokens=True)
            c_text = tokenizer.decode(a[0], skip_special_tokens=True)
            c_text_tokens = tokenizer(c_text, return_tensors="pt").input_ids

            sample_outputs = model.generate(
                c_text_tokens,
                # attention_mask=a[1],
                do_sample=False,
                top_k=50,
                max_length=512,
                top_p=0.70,
                temperature=1,
                num_return_sequences=0,
            )

            # decode the predicted tokens into texts
            # context = tokenizer.decode(context, skip_special_tokens=True)
            pred_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
            a = 1


if __name__ == "__main__":
    run()
