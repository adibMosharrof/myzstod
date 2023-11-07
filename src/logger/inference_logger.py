import utils
import numpy as np
import pandas as pd


class InferenceLogger:
    def __init__(self, csv_path, tokenizer, metric_manager):
        self.all_input_texts = []
        self.all_labels = []
        self.all_preds = []
        self.all_gleu_scores = []
        self.csv_path = csv_path
        self.concat_labels = None
        self.concat_preds = None
        self.tokenizer = tokenizer
        self.metric_manager = metric_manager

    def add_batch(self, input_tokens, label_tokens, pred_tokens):
        input_texts = self.tokenizer.batch_decode(
            input_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        preds = self.tokenizer.batch_decode(
            pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        labels = self.tokenizer.batch_decode(
            label_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        self.all_input_texts.append(input_texts)
        self.all_labels.append(labels)
        self.all_preds.append(preds)
        self.all_gleu_scores.append(
            self.metric_manager.compute_single_row(preds, labels)
        )

    def write_csv(self):
        self.concat_labels = np.concatenate(self.all_labels, axis=0)
        self.concat_preds = np.concatenate(self.all_preds, axis=0)
        concat_input_texts = np.concatenate(self.all_input_texts, axis=0)
        concat_gleu_scores = np.concatenate(self.all_gleu_scores, axis=0)

        df = pd.DataFrame(
            {
                "input_texts": concat_input_texts,
                "target_text": self.concat_labels,
                "pred_text": self.concat_preds,
                "gleu_score": concat_gleu_scores,
            }
        )

        df.to_csv(self.csv_path, index=False, encoding="utf-8")
