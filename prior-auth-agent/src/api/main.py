import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.api.csv_import import import_csv_requests
from src.db.database import SessionLocal, engine, get_db, init_db
from src.db.models import AgentAnalysis, HumanDecision, PriorAuthRequest
from src.db.preanalyze import preanalyze_pending_requests
from src.db.seed import seed_synthetic_requests
from src.rag.retriever import get_document_count, warmup_retriever

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_PATH = ROOT_DIR / "frontend" / "index.html"
KNOWLEDGE_BASE_PATH = Path(os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base"))
if not KNOWLEDGE_BASE_PATH.is_absolute():
    KNOWLEDGE_BASE_PATH = ROOT_DIR / KNOWLEDGE_BASE_PATH

# Industry average: ~20 min manual clinical review per prior auth request (AMA/MGMA estimates)
MANUAL_REVIEW_MINUTES = float(os.getenv("MANUAL_REVIEW_MINUTES", "20"))

app = FastAPI(title="ClearAuth", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_graph = build_graph()
pending_states: dict[str, dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    patient_age: int = Field(ge=1, le=120)
    patient_sex: str
    diagnosis_code: str
    diagnosis_label: str
    procedure_code: Optional[str] = ""
    procedure_label: str
    clinical_notes: Optional[str] = ""
    urgency: str


class DecisionRequest(BaseModel):
    request_id: str
    analysis_id: str
    human_action: str
    override_reason: Optional[str] = None


def _migrate_schema() -> None:
    """Add new columns to existing SQLite DB if missing."""
    cols = {
        "is_synthetic": "INTEGER DEFAULT 0",
        "expected_decision": "TEXT",
        "sample_index": "INTEGER",
    }
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(prior_auth_requests)"))}
        for col, typedef in cols.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE prior_auth_requests ADD COLUMN {col} {typedef}"))
        conn.commit()


def _check_duplicate(db: Session, age: int, sex: str, procedure_label: str) -> bool:
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    existing = (
        db.query(PriorAuthRequest)
        .filter(
            PriorAuthRequest.patient_age == age,
            PriorAuthRequest.patient_sex == sex,
            PriorAuthRequest.procedure_label == procedure_label,
            PriorAuthRequest.created_at >= cutoff,
        )
        .first()
    )
    return existing is not None


def _map_final_status(human_action: str) -> str:
    action = human_action.lower()
    if action == "approve":
        return "approved"
    if action in ("reject", "override"):
        return "rejected" if action == "reject" else "overridden"
    return "escalated"


def _map_request_status(human_action: str) -> str:
    action = human_action.lower()
    if action == "approve":
        return "approved"
    if action in ("reject", "override"):
        return "rejected" if action == "reject" else "overridden"
    return "escalated"


def _request_to_dict(req: PriorAuthRequest, analysis=None, decision=None) -> dict:
    sex = "M" if req.patient_sex.upper().startswith("M") else "F"
    item = {
        "request_id": req.id,
        "created_at": req.created_at.isoformat() if req.created_at else None,
        "patient_age": req.patient_age,
        "patient_sex": req.patient_sex,
        "patient_label": f"{req.patient_age}{sex}",
        "diagnosis_code": req.diagnosis_code,
        "diagnosis_label": req.diagnosis_label,
        "procedure_code": req.procedure_code,
        "procedure_label": req.procedure_label,
        "clinical_notes": req.clinical_notes,
        "urgency": req.urgency,
        "status": req.status,
        "is_synthetic": bool(req.is_synthetic),
        "expected_decision": req.expected_decision,
        "sample_index": req.sample_index,
        "ai_decision": analysis.ai_decision if analysis else None,
        "confidence_score": analysis.confidence_score if analysis else None,
        "final_status": decision.final_status if decision else req.status,
    }
    return item


def _run_agent_analysis(db_request: PriorAuthRequest) -> dict[str, Any]:
    start = time.time()
    request_id = db_request.id
    analysis_id = str(uuid.uuid4())
    urgency = db_request.urgency.lower()
    force_escalate = urgency == "emergent"

    initial_state: AgentState = {
        "request_id": request_id,
        "patient_age": db_request.patient_age,
        "patient_sex": db_request.patient_sex,
        "diagnosis_code": db_request.diagnosis_code,
        "diagnosis_label": db_request.diagnosis_label,
        "procedure_code": db_request.procedure_code or "",
        "procedure_label": db_request.procedure_label,
        "clinical_notes": db_request.clinical_notes or "",
        "urgency": urgency,
        "parsed_query": "",
        "policy_citations": [],
        "reasoning_steps": [],
        "ai_decision": "ESCALATE",
        "confidence_score": 0.0,
        "reasoning_trace": [],
        "error": None,
        "processing_start": start,
        "raw_llm_response": None,
        "processing_time_ms": None,
    }

    config = {"configurable": {"thread_id": request_id}}
    result = agent_graph.invoke(initial_state, config)
    processing_time_ms = int((time.time() - start) * 1000)

    ai_decision = result.get("ai_decision", "ESCALATE")
    confidence = float(result.get("confidence_score", 0.0))

    if force_escalate:
        ai_decision = "ESCALATE"
    if confidence < 0.60:
        ai_decision = "ESCALATE"

    low_confidence_forced = float(result.get("confidence_score", 0.0)) < 0.60

    return {
        "request_id": request_id,
        "analysis_id": analysis_id,
        "ai_decision": ai_decision,
        "confidence_score": confidence,
        "reasoning_trace": result.get("reasoning_trace", []),
        "policy_citations": result.get("policy_citations", []),
        "processing_time_ms": processing_time_ms,
        "low_confidence_forced": low_confidence_forced,
        "no_policy_found": result.get("error") == "NO_POLICY_FOUND",
        "emergent_escalated": force_escalate,
        "agent_error": result.get("error"),
        "raw_llm_response": result.get("raw_llm_response"),
        "config": config,
        "result": result,
    }


def _persist_analysis(db: Session, request_id: str, outcome: dict[str, Any]) -> AgentAnalysis:
    analysis = AgentAnalysis(
        id=outcome["analysis_id"],
        request_id=request_id,
        ai_decision=outcome["ai_decision"],
        confidence_score=outcome["confidence_score"],
        reasoning_trace=json.dumps(outcome["reasoning_trace"]),
        policy_citations=json.dumps(outcome["policy_citations"]),
        raw_llm_response=outcome.get("raw_llm_response"),
        processing_time_ms=outcome["processing_time_ms"],
    )
    db.add(analysis)
    db.commit()

    pending_states[request_id] = {
        "analysis_id": outcome["analysis_id"],
        "config": outcome["config"],
        "state": outcome["result"],
    }
    return analysis


def _analyze_and_persist(db: Session, db_request: PriorAuthRequest) -> dict[str, Any]:
    outcome = _run_agent_analysis(db_request)
    _persist_analysis(db, db_request.id, outcome)
    response = {k: v for k, v in outcome.items() if k not in ("config", "result", "raw_llm_response")}
    return response


def _startup_background() -> None:
    db = SessionLocal()
    try:
        warmup_retriever()
        seeded = seed_synthetic_requests(db)
        if seeded:
            print(f"Seeded {seeded} synthetic requests")
        preanalyze_pending_requests(db, _analyze_and_persist)
    except Exception as exc:
        logger.exception("Startup background task failed: %s", exc)
    finally:
        db.close()


def _analyze_imported_background(request_ids: list[str]) -> None:
    db = SessionLocal()
    try:
        for rid in request_ids:
            req = db.query(PriorAuthRequest).filter(PriorAuthRequest.id == rid).first()
            if not req:
                continue
            existing = db.query(AgentAnalysis).filter(AgentAnalysis.request_id == rid).first()
            if existing:
                continue
            try:
                _analyze_and_persist(db, req)
            except Exception as exc:
                logger.exception("Failed to analyze imported request %s: %s", rid, exc)
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    _migrate_schema()
    threading.Thread(target=_startup_background, daemon=True).start()


@app.get("/")
def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(FRONTEND_PATH)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "chroma_documents": get_document_count()}


@app.get("/api/policy/pdf/{filename}")
def serve_policy_pdf(filename: str):
    """Serve a policy PDF from the knowledge base (filename only, no path traversal)."""
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_path = (KNOWLEDGE_BASE_PATH / safe_name).resolve()
    kb_root = KNOWLEDGE_BASE_PATH.resolve()
    if not str(pdf_path).startswith(str(kb_root)):
        raise HTTPException(status_code=404, detail="Policy document not found")
    if not pdf_path.is_file():
        raise HTTPException(status_code=404, detail="Policy document not found")

    return FileResponse(pdf_path, media_type="application/pdf", filename=safe_name)


@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    decision_count = db.query(HumanDecision).count()
    hours_saved = round(decision_count * MANUAL_REVIEW_MINUTES / 60, 1)
    return {
        "decisions_count": decision_count,
        "manual_review_minutes": MANUAL_REVIEW_MINUTES,
        "hours_saved": hours_saved,
    }


@app.post("/api/import/csv")
async def import_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    content = await file.read()
    try:
        created = import_csv_requests(db, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request_ids = [r.id for r in created]
    if request_ids:
        threading.Thread(target=_analyze_imported_background, args=(request_ids,), daemon=True).start()

    return {
        "imported": len(created),
        "request_ids": request_ids,
        "message": f"Imported {len(created)} request(s). AI analysis running in background.",
    }


@app.get("/api/pending")
def get_pending(db: Session = Depends(get_db)):
    requests = (
        db.query(PriorAuthRequest)
        .filter(PriorAuthRequest.status == "pending")
        .order_by(PriorAuthRequest.sample_index.asc().nullslast(), PriorAuthRequest.created_at.asc())
        .all()
    )
    items = []
    for req in requests:
        latest_analysis = (
            db.query(AgentAnalysis)
            .filter(AgentAnalysis.request_id == req.id)
            .order_by(AgentAnalysis.created_at.desc())
            .first()
        )
        items.append(_request_to_dict(req, latest_analysis))
    return {"requests": items, "total": len(items)}


@app.get("/api/requests/{request_id}")
def get_request(request_id: str, db: Session = Depends(get_db)):
    req = db.query(PriorAuthRequest).filter(PriorAuthRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    analysis = (
        db.query(AgentAnalysis)
        .filter(AgentAnalysis.request_id == request_id)
        .order_by(AgentAnalysis.created_at.desc())
        .first()
    )
    decision = (
        db.query(HumanDecision)
        .filter(HumanDecision.request_id == request_id)
        .order_by(HumanDecision.created_at.desc())
        .first()
    )

    data = _request_to_dict(req, analysis, decision)
    if analysis:
        data["analysis"] = {
            "analysis_id": analysis.id,
            "ai_decision": analysis.ai_decision,
            "confidence_score": analysis.confidence_score,
            "reasoning_trace": json.loads(analysis.reasoning_trace),
            "policy_citations": json.loads(analysis.policy_citations),
            "processing_time_ms": analysis.processing_time_ms,
            "low_confidence_forced": analysis.confidence_score < 0.60,
        }
    if decision:
        data["decision"] = {
            "decision_id": decision.id,
            "human_action": decision.human_action,
            "override_reason": decision.override_reason,
            "reviewer_id": decision.reviewer_id,
            "final_status": decision.final_status,
            "created_at": decision.created_at.isoformat() if decision.created_at else None,
        }
    return data


@app.post("/api/requests/{request_id}/analyze")
def analyze_existing_request(request_id: str, db: Session = Depends(get_db)):
    """Return cached analysis only — analysis runs at creation or startup."""
    req = db.query(PriorAuthRequest).filter(PriorAuthRequest.id == request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")

    existing_analysis = (
        db.query(AgentAnalysis)
        .filter(AgentAnalysis.request_id == request_id)
        .order_by(AgentAnalysis.created_at.desc())
        .first()
    )
    if not existing_analysis:
        raise HTTPException(status_code=202, detail="Analysis not ready yet — still processing")

    return {
        "request_id": request_id,
        "analysis_id": existing_analysis.id,
        "ai_decision": existing_analysis.ai_decision,
        "confidence_score": existing_analysis.confidence_score,
        "reasoning_trace": json.loads(existing_analysis.reasoning_trace),
        "policy_citations": json.loads(existing_analysis.policy_citations),
        "processing_time_ms": existing_analysis.processing_time_ms,
        "low_confidence_forced": existing_analysis.confidence_score < 0.60,
        "cached": True,
    }


@app.post("/api/analyze")
def analyze_request(payload: AnalyzeRequest, db: Session = Depends(get_db)):
    duplicate_warning = _check_duplicate(
        db, payload.patient_age, payload.patient_sex, payload.procedure_label
    )

    request_id = str(uuid.uuid4())
    db_request = PriorAuthRequest(
        id=request_id,
        patient_age=payload.patient_age,
        patient_sex=payload.patient_sex,
        diagnosis_code=payload.diagnosis_code,
        diagnosis_label=payload.diagnosis_label,
        procedure_code=payload.procedure_code or "",
        procedure_label=payload.procedure_label,
        clinical_notes=payload.clinical_notes or "",
        urgency=payload.urgency.lower(),
        status="pending",
        is_synthetic=False,
    )
    db.add(db_request)
    db.commit()

    try:
        outcome = _analyze_and_persist(db, db_request)
    except Exception as exc:
        raise HTTPException(status_code=504, detail=f"Agent failed: {exc}") from exc

    response = {
        "request_id": request_id,
        **outcome,
        "duplicate_warning": duplicate_warning,
    }
    return response


@app.post("/api/decision")
def submit_decision(payload: DecisionRequest, db: Session = Depends(get_db)):
    if payload.human_action.lower() in ("reject", "override") and not (
        payload.override_reason and payload.override_reason.strip()
    ):
        raise HTTPException(status_code=400, detail="Reason is required for reject/override")

    db_request = db.query(PriorAuthRequest).filter(PriorAuthRequest.id == payload.request_id).first()
    if not db_request:
        raise HTTPException(status_code=404, detail="Request not found")

    analysis = (
        db.query(AgentAnalysis)
        .filter(
            AgentAnalysis.id == payload.analysis_id,
            AgentAnalysis.request_id == payload.request_id,
        )
        .first()
    )
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    final_status = _map_final_status(payload.human_action)
    request_status = _map_request_status(payload.human_action)

    decision_id = str(uuid.uuid4())
    human_decision = HumanDecision(
        id=decision_id,
        request_id=payload.request_id,
        analysis_id=payload.analysis_id,
        human_action=payload.human_action.lower(),
        override_reason=payload.override_reason,
        reviewer_id="demo-reviewer",
        final_status=final_status,
    )
    db.add(human_decision)
    db_request.status = request_status
    db.commit()

    # Human decision is persisted above; LangGraph log_decision is a no-op.
    # Resume graph in background so the API responds immediately.
    pending = pending_states.pop(payload.request_id, None)
    if pending:
        def _resume_graph() -> None:
            try:
                agent_graph.invoke(None, pending["config"])
            except Exception:
                pass

        threading.Thread(target=_resume_graph, daemon=True).start()

    return {
        "decision_id": decision_id,
        "final_status": final_status,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    requests = (
        db.query(PriorAuthRequest)
        .filter(PriorAuthRequest.status != "pending")
        .order_by(PriorAuthRequest.created_at.desc())
        .all()
    )
    items = []
    for req in requests:
        latest_analysis = (
            db.query(AgentAnalysis)
            .filter(AgentAnalysis.request_id == req.id)
            .order_by(AgentAnalysis.created_at.desc())
            .first()
        )
        latest_decision = (
            db.query(HumanDecision)
            .filter(HumanDecision.request_id == req.id)
            .order_by(HumanDecision.created_at.desc())
            .first()
        )
        items.append(_request_to_dict(req, latest_analysis, latest_decision))
    return {"requests": items}


@app.get("/api/history/{request_id}")
def get_history_detail(request_id: str, db: Session = Depends(get_db)):
    return get_request(request_id, db)
