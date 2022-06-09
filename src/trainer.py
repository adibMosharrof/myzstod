from transformers import Trainer, TrainingArguments


class MyTrainer(Trainer):
    def __init__(self, model, training_args, data_collator, dm):
        super().__init__(
            model=model,
            args=training_args,
            # data_collator=data_collator,
            # train_dataset=dm.datasets["train"],
            # eval_dataset=dm.datasets["dev"],
        )
        self.dm = dm

    def training_step(self, model, inputs):
        self.shared_step(model, inputs, "train")

    def prediction_step(self, model, inputs):
        self.shared_step(inputs)

    def shared_step(self, model, inputs, step="train"):
        a = 1

    def get_train_dataloader(self):
        return self.dm.train_dataloader()

    def get_eval_dataloader(self):
        return self.dm.val_dataloader()

    def get_test_dataloader(self):
        return self.dm.test_dataloader()
