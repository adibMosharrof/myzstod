from dataclasses import dataclass
from typing import Optional


@dataclass
class BertScoreData:
    precision: float
    recall: float
    f1: float

    def __str__(self):
        return f"Bert Score: precision {self.precision:.4f}, recall {self.recall:.4f}, f1 {self.f1:.4f}"


@dataclass
class InferenceLogData:
    input_text: str
    label: str
    pred: str
    gleu_score: Optional[float] = None
    bert_score_data: Optional[BertScoreData] = None


@dataclass
class ServiceCallInferenceLogData(InferenceLogData):
    is_api_call: Optional[int] = None
