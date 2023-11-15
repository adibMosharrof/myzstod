import uuid
import evaluate
import numpy as np

from logger.inference_logger import InferenceLogger


class NlgMetricManager:
    def __init__(self, logger):
        self.google_bleu = evaluate.load("google_bleu", experiment_id=str(uuid.uuid4()))
        self.bert_score = evaluate.load("bertscore", experiment_id=str(uuid.uuid4()))
        self.bert_score_model = "distilbert-base-uncased"
        self.logger = logger

    def compute_gleu(self, inf_logger: InferenceLogger):
        gleu_labels = np.expand_dims(inf_logger.concat_labels, axis=1)
        result = self.google_bleu.compute(
            predictions=inf_logger.concat_preds, references=gleu_labels
        )
        score_str = f"GLEU score: {result['google_bleu']:.4f}"
        self.logger.info(score_str)
        print(score_str)

    def compute_bert_score(self, inf_logger: InferenceLogger):
        result = self.bert_score.compute(
            predictions=inf_logger.concat_preds,
            references=inf_logger.concat_labels,
            model_type=self.bert_score_model,
        )
        avg_precision = np.mean(result["precision"])
        avg_recall = np.mean(result["recall"])
        avg_f1 = np.mean(result["f1"])
        score_str = f"BERT score: precision {avg_precision:.4f}, recall {avg_recall:.4f}, f1 {avg_f1:.4f}"
        self.logger.info(score_str)
        print(score_str)

    def compute_metrics(self, inf_logger: InferenceLogger):
        gleu_labels = np.expand_dims(inf_logger.concat_labels, axis=1)
        result = self.google_bleu.compute(
            predictions=inf_logger.concat_preds, references=gleu_labels
        )
        score_str = f"GLEU score: {result['google_bleu']:.4f}"
        self.logger.info(score_str)
        print(score_str)
        bert_score_str = self.compute_bert_score(inf_logger)
        self.logger.info(bert_score_str)
        print(bert_score_str)

    def compute_single_row(self, preds, labels):
        bleu_result = self.google_bleu.compute(
            predictions=[preds], references=[[labels]]
        )
        bert_score_result = self.bert_score.compute(
            predictions=[preds],
            references=[labels],
            model_type=self.bert_score_model,
        )
        avg_precision = np.mean(bert_score_result["precision"])
        avg_recall = np.mean(bert_score_result["recall"])
        avg_f1 = np.mean(bert_score_result["f1"])
        return (
            round(bleu_result["google_bleu"], 4),
            round(avg_precision, 4),
            round(avg_recall, 4),
            round(avg_f1, 4),
        )
