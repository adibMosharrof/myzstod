from dataclasses import dataclass
from typing import Optional

from my_enums import TurnRowType


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
class ApiCallInferenceLogData(InferenceLogData):
    turn_row_type: Optional[int] = None
    complete_api_call: Optional[int] = None
    api_call_method: Optional[float] = None
    api_call_param_names: Optional[float] = None
    api_call_param_values: Optional[float] = None
    response_gleu: Optional[float] = None
    response_bleu: Optional[float] = None
    api_call_invoke: Optional[float] = None
    is_retrieval: Optional[int] = 0
    is_slot_fill: Optional[int] = 0
    dialog_id: Optional[int] = None
    turn_id: Optional[int] = None
    domains: Optional[list[str]] = None

    def update(self, updated_data: "ApiCallInferenceLogData"):
        self.turn_row_type = updated_data.turn_row_type
        self.complete_api_call = updated_data.complete_api_call
        self.api_call_method = updated_data.api_call_method
        self.api_call_param_names = updated_data.api_call_param_names
        self.api_call_param_values = updated_data.api_call_param_values
        self.response_gleu = updated_data.response_gleu
        self.response_bleu = updated_data.response_bleu
        self.api_call_invoke = updated_data.api_call_invoke
        self.is_retrieval = updated_data.is_retrieval
        self.is_slot_fill = updated_data.is_slot_fill
        self.domains = updated_data.domains


@dataclass
class KetodInferenceLogData(ApiCallInferenceLogData):
    complete_kb_call: Optional[int] = None
    ke_method: Optional[float] = None
    ke_params: Optional[float] = None
    ke_api_call_invoke: Optional[float] = None
