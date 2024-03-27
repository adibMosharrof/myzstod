import utils
import numpy as np
import pandas as pd


class InferenceLogger:
    def __init__(self, tokenizer, metric_manager):
        self.all_input_texts = []
        self.all_labels = []
        self.all_preds = []
        self.all_gleu_scores = []
        self.all_bert_score_precision = []
        self.all_bert_score_recall = []
        self.all_bert_score_f1 = []
        self.concat_labels = None
        self.concat_preds = None
        self.tokenizer = tokenizer
        self.metric_manager = metric_manager

    def add_batch(self, input_tokens, label_tokens, pred_tokens, api_calls):
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

        for p, l, s in zip(preds, labels, api_calls):
            (
                gleu_score,
                b_precision,
                b_recall,
                b_f1,
            ) = self.metric_manager.compute_single_row(p, l)
            self.all_gleu_scores.append(gleu_score)
            self.all_bert_score_precision.append(b_precision)
            self.all_bert_score_recall.append(b_recall)
            self.all_bert_score_f1.append(b_f1)

    def write_csv(self, csv_path):
        self.concat_labels = np.concatenate(self.all_labels, axis=0)
        self.concat_preds = np.concatenate(self.all_preds, axis=0)
        concat_input_texts = np.concatenate(self.all_input_texts, axis=0)

        df = pd.DataFrame(
            {
                "input_texts": concat_input_texts,
                "target_text": self.concat_labels,
                "pred_text": self.concat_preds,
                "gleu_score": self.all_gleu_scores,
                "bs_precision": self.all_bert_score_precision,
                "bs_recall": self.all_bert_score_recall,
                "bs_f1": self.all_bert_score_f1,
            }
        )

        df.to_csv(csv_path, index=False, encoding="utf-8")
