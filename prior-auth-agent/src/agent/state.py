from typing import List, Optional, TypedDict


class PolicyCitation(TypedDict):
    source: str
    source_file: str
    page_number: Optional[int]
    text: str
    relevance_score: float


class ReasoningStep(TypedDict):
    step: int
    label: str
    detail: str


class AgentState(TypedDict):
    # Input
    request_id: str
    patient_age: int
    patient_sex: str
    diagnosis_code: str
    diagnosis_label: str
    procedure_code: str
    procedure_label: str
    clinical_notes: str
    urgency: str

    # Intermediate
    parsed_query: str
    policy_citations: List[PolicyCitation]
    reasoning_steps: List[ReasoningStep]

    # Output
    ai_decision: str
    confidence_score: float
    reasoning_trace: List[ReasoningStep]

    # Control
    error: Optional[str]
    processing_start: float
    raw_llm_response: Optional[str]
    processing_time_ms: Optional[int]
