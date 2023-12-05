from dataclasses import dataclass
from typing import Optional


@dataclass
class BertScoreData:
    precision: float
    recall: float
    f1: float

    def get_row_str(self):
        return f"{self.precision:.4f}, {self.recall:.4f}, {self.f1:.4f}"

    def __str__(self):
        return f"Bert Score: precision {self.precision:.4f}, recall {self.recall:.4f}, f1 {self.f1:.4f}"


@dataclass
class InferenceLogData:
    input_text: str
    label: str
    pred: str


@dataclass
class ServiceCallInferenceLogData(InferenceLogData):
    is_api_call: Optional[int] = None
    complete_api_call: Optional[int] = None
    api_call_method: Optional[float] = None
    api_call_params: Optional[float] = None
    response_gleu: Optional[float] = None
    response_bertscore: Optional[BertScoreData] = None
