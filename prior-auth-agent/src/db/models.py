import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from src.db.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


class PriorAuthRequest(Base):
    __tablename__ = "prior_auth_requests"

    id = Column(String, primary_key=True, default=_new_uuid)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    patient_age = Column(Integer, nullable=False)
    patient_sex = Column(String, nullable=False)
    diagnosis_code = Column(String, nullable=False)
    diagnosis_label = Column(String, nullable=False)
    procedure_code = Column(String, nullable=True)
    procedure_label = Column(String, nullable=False)
    clinical_notes = Column(Text, nullable=True)
    urgency = Column(String, nullable=False)
    status = Column(String, default="pending", nullable=False)
    is_synthetic = Column(Integer, default=0, nullable=False)
    expected_decision = Column(String, nullable=True)
    sample_index = Column(Integer, nullable=True)

    analyses = relationship("AgentAnalysis", back_populates="request")
    decisions = relationship("HumanDecision", back_populates="request")


class AgentAnalysis(Base):
    __tablename__ = "agent_analyses"

    id = Column(String, primary_key=True, default=_new_uuid)
    request_id = Column(String, ForeignKey("prior_auth_requests.id"), nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    ai_decision = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    reasoning_trace = Column(Text, nullable=False)
    policy_citations = Column(Text, nullable=False)
    raw_llm_response = Column(Text, nullable=True)
    processing_time_ms = Column(Integer, nullable=False)

    request = relationship("PriorAuthRequest", back_populates="analyses")
    human_decisions = relationship("HumanDecision", back_populates="analysis")


class HumanDecision(Base):
    __tablename__ = "human_decisions"

    id = Column(String, primary_key=True, default=_new_uuid)
    request_id = Column(String, ForeignKey("prior_auth_requests.id"), nullable=False)
    analysis_id = Column(String, ForeignKey("agent_analyses.id"), nullable=False)
    created_at = Column(DateTime, default=_utcnow, nullable=False)
    human_action = Column(String, nullable=False)
    override_reason = Column(Text, nullable=True)
    reviewer_id = Column(String, default="demo-reviewer", nullable=False)
    final_status = Column(String, nullable=False)

    request = relationship("PriorAuthRequest", back_populates="decisions")
    analysis = relationship("AgentAnalysis", back_populates="human_decisions")
