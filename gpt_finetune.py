#!pip install transformers==4.8.2

import random
import re

import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

## Define class and functions
# --------

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, txt_list, label_list, tokenizer, max_length):
        # define variables
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        map_label = {0: "negative", 4: "positive"}
        # iterate through the dataset
        for txt, label in zip(txt_list, label_list):
            # prepare the text
            prep_txt = f"<|startoftext|>Tweet: {txt}\nSentiment: {map_label[label]}<|endoftext|>"
            # tokenize
            encodings_dict = tokenizer(
                prep_txt, truncation=True, max_length=max_length, padding="max_length"
            )
            # append to list
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))
            self.labels.append(map_label[label])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


# Data load function
def load_sentiment_dataset(tokenizer):
    # load dataset and sample 10k reviews.
    file_path = "training.1600000.processed.noemoticon.csv"
    df = pd.read_csv(file_path, encoding="ISO-8859-1", header=None)
    df = df[[0, 5]]
    df.columns = ["label", "text"]
    df = df.sample(10000, random_state=1)

    # divide into test and train
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        shuffle=True,
        test_size=0.05,
        random_state=1,
        stratify=df["label"],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, shuffle=True, test_size=0.1, random_state=1, stratify=y_train
    )
    # format into SentimentDataset class
    train_dataset = SentimentDataset(X_train, y_train, tokenizer, max_length=512)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer, max_length=512)

    # return
    return train_dataset, val_dataset, (X_test, y_test)


## Load model and data
# --------

# set model name
model_name = "gpt2"
# seed
torch.manual_seed(42)

# load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(
    model_name,
    bos_token="<|startoftext|>",
    eos_token="<|endoftext|>",
    pad_token="<|pad|>",
)
model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
model.resize_token_embeddings(len(tokenizer))

# prepare and load dataset
train_dataset, val_dataset, test_dataset = load_sentiment_dataset(tokenizer)

## Train
# --------
# creating training arguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=2,
    logging_steps=10,
    load_best_model_at_end=True,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="logs",
)

# start training
Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda data: {
        "input_ids": torch.stack([f[0] for f in data]),
        "attention_mask": torch.stack([f[1] for f in data]),
        "labels": torch.stack([f[0] for f in data]),
    },
).train()

## Test
# ----------

# set the model to eval mode
_ = model.eval()

# run model inference on all test data
original_label, predicted_label, original_text, predicted_text = [], [], [], []
map_label = {0: "negative", 4: "positive"}
# iter over all of the test data
for text, label in tqdm(zip(test_dataset[0], test_dataset[1])):
    # create prompt (in compliance with the one used during training)
    prompt = f"<|startoftext|>Tweet: {text}\nSentiment:"
    # generate tokens
    generated = tokenizer(f"{prompt}", return_tensors="pt").input_ids.cuda()
    # perform prediction
    sample_outputs = model.generate(
        generated,
        do_sample=False,
        top_k=50,
        max_length=512,
        top_p=0.90,
        temperature=0,
        num_return_sequences=0,
    )
    # decode the predicted tokens into texts
    pred_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    # extract the predicted sentiment
    try:
        pred_sentiment = re.findall("\nSentiment: (.*)", pred_text)[-1]
    except:
        pred_sentiment = "None"
    # append results
    original_label.append(map_label[label])
    predicted_label.append(pred_sentiment)
    original_text.append(text)
    predicted_text.append(pred_text)

# transform result into dataframe
df = pd.DataFrame(
    {
        "original_text": original_text,
        "predicted_label": predicted_label,
        "original_label": original_label,
        "predicted_text": predicted_text,
    }
)

# predict the accuracy
print(f1_score(original_label, predicted_label, average="macro"))
