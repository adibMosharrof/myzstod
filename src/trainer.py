# import pandas as pd
import hydra
from omegaconf import DictConfig
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


class SimpleTODTrainer:
    def __init__(
        self,
        model_name: str = None,
        epochs: int = 2,
        train_batch_size: int = 30,
        eval_batch_size: int = 30,
        output_dir: str = "results",
        logging_dir: str = "logs",
        logging_steps: int = 10,
    ) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps

    def run(self):

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>",
        )
        # special_tokens = torch.tensor(SpecialTokens.list(), device=torch.device("cuda"))
        special_tokens = SpecialTokens.list()
        tokenizer.add_tokens(special_tokens, special_tokens=True)
        model = GPT2LMHeadModel.from_pretrained(self.model_name)
        model.resize_token_embeddings(len(tokenizer))
        model = model.cuda()

        data_root = "/localdisk0/adibm/data/dstc8-schema-guided-dialogue/processed_out/"
        dm = SimpleTodDataModule(tokenizer=tokenizer, data_root=data_root)
        dm.setup()
        self.train(model, dm)
        # return self.test(dm, tokenizer)

    def compute_metrics(self, preds):
        a = 1

    def train(self, model, dm):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            logging_steps=self.logging_steps,
            load_best_model_at_end=True,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=self.logging_dir,
        )

        # start training
        Trainer(
            model=model,
            args=training_args,
            train_dataset=dm.datasets["train"],
            eval_dataset=dm.datasets["dev"],
            compute_metrics=self.compute_metrics,
            data_collator=dm.my_collate,
            # data_collator=lambda data: {
            #     "input_ids": torch.stack([f[0]["input_ids"] for f in data]),
            #     "attention_mask": torch.stack([f[0]["attention_mask"] for f in data]),
            #     "labels": torch.stack([f[1]["input_ids"] for f in data]),
            # },
        ).train()

    def test(self, dm, tokenizer):
        model = GPT2LMHeadModel.from_pretrained("results/checkpoint-178")

        # for contexts_batch, targets_batch in tqdm(dm.test_dataloader()):
        for row in tqdm(dm.test_dataloader()):
            # (_,c_id), (_,c_am), (_,t_id), (_,t_am) = row.items()
            (_, c_id), (_, c_am), (_, l_id), (_, c_texts), (_, t_texts) = row.items()
            # for a in zip(c_id, c_am, t_id, t_am):
            for a in zip(c_id, c_am, l_id, c_texts, t_texts):
                t_text = a[4]
                c_text = a[3]
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
                pred_text = tokenizer.decode(
                    sample_outputs[0], skip_special_tokens=False
                )
                a = 1


@hydra.main(config_path="../config/trainer/", config_name="simple_tod_trainer")
def hydra_start(cfg: DictConfig) -> None:
    stt = SimpleTODTrainer(
        epochs=cfg.epochs,
        model_name=cfg.model_name,
        train_batch_size=cfg.train_batch_size,
        eval_batch_size=cfg.eval_batch_size,
        output_dir=cfg.output_dir,
        logging_dir=cfg.logging_dir,
        logging_steps=cfg.logging_steps,
    )
    stt.run()


if __name__ == "__main__":
    hydra_start()
