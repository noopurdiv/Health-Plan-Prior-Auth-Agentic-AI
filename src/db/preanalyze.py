import logging

from sqlalchemy.orm import Session

from src.db.models import AgentAnalysis, PriorAuthRequest

logger = logging.getLogger(__name__)


def preanalyze_pending_requests(db: Session, run_analysis_fn) -> int:
    """Run AI analysis for all pending requests that lack an analysis record."""
    pending = (
        db.query(PriorAuthRequest)
        .filter(PriorAuthRequest.status == "pending")
        .order_by(PriorAuthRequest.sample_index.asc().nullslast())
        .all()
    )

    analyzed = 0
    for req in pending:
        existing = (
            db.query(AgentAnalysis)
            .filter(AgentAnalysis.request_id == req.id)
            .first()
        )
        if existing:
            continue

        label = f"{req.patient_age}{req.patient_sex[0]} — {req.procedure_label[:40]}"
        logger.info("Pre-analyzing: %s", label)
        print(f"Pre-analyzing ({analyzed + 1}): {label}")

        try:
            run_analysis_fn(db, req)
            analyzed += 1
        except Exception as exc:
            logger.exception("Failed to pre-analyze %s: %s", req.id, exc)
            print(f"  Failed: {exc}")

    if analyzed:
        print(f"Pre-analyzed {analyzed} pending request(s)")
    return analyzed
