#!/usr/bin/env python3
"""Evaluate AI decisions against expected labels on synthetic benchmark cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from src.db.database import SessionLocal, init_db  # noqa: E402
from src.db.models import AgentAnalysis, PriorAuthRequest  # noqa: E402

SAMPLE_PATH = ROOT / "synthetic_data" / "sample_requests.json"


def decisions_match(expected: str | None, actual: str | None) -> bool:
    """Return True when AI decision exactly matches the expected benchmark label."""
    if not expected or not actual:
        return False
    return expected.strip().upper() == actual.strip().upper()


def compute_metrics(results: list[dict]) -> dict:
    """Summarize benchmark results."""
    evaluated = [r for r in results if r.get("ai_decision")]
    total = len(evaluated)
    matches = sum(1 for r in evaluated if r.get("match"))
    missing = len(results) - total

    by_expected: dict[str, dict[str, int]] = {}
    for row in evaluated:
        exp = row.get("expected_decision", "UNKNOWN")
        by_expected.setdefault(exp, {"total": 0, "correct": 0})
        by_expected[exp]["total"] += 1
        if row.get("match"):
            by_expected[exp]["correct"] += 1

    accuracy = round(matches / total * 100, 1) if total else 0.0
    return {
        "total_cases": len(results),
        "evaluated": total,
        "missing_analysis": missing,
        "matches": matches,
        "accuracy_pct": accuracy,
        "by_expected": by_expected,
    }


def load_benchmark_rows(db: Session) -> list[dict]:
    """Load synthetic requests with latest AI analysis from the database."""
    requests = (
        db.query(PriorAuthRequest)
        .filter(
            PriorAuthRequest.is_synthetic == True,  # noqa: E712
            PriorAuthRequest.expected_decision.isnot(None),
        )
        .order_by(PriorAuthRequest.sample_index.asc().nullslast())
        .all()
    )

    rows: list[dict] = []
    for req in requests:
        analysis = (
            db.query(AgentAnalysis)
            .filter(AgentAnalysis.request_id == req.id)
            .order_by(AgentAnalysis.created_at.desc())
            .first()
        )
        ai_decision = analysis.ai_decision if analysis else None
        rows.append(
            {
                "sample_index": req.sample_index,
                "procedure_label": req.procedure_label,
                "patient_label": f"{req.patient_age}{req.patient_sex[0]}",
                "expected_decision": req.expected_decision,
                "ai_decision": ai_decision,
                "confidence_score": analysis.confidence_score if analysis else None,
                "match": decisions_match(req.expected_decision, ai_decision),
            }
        )
    return rows


def load_expected_from_json() -> list[dict]:
    """Fallback labels from sample_requests.json when DB is empty."""
    if not SAMPLE_PATH.exists():
        return []
    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return [
        {
            "sample_index": i,
            "procedure_label": s.get("procedure_label", ""),
            "patient_label": f"{s['patient_age']}{s['patient_sex'][0]}",
            "expected_decision": s.get("expected_decision"),
            "ai_decision": None,
            "confidence_score": None,
            "match": False,
        }
        for i, s in enumerate(samples)
    ]


def format_report(results: list[dict], metrics: dict) -> str:
    lines = [
        "ClearAuth Synthetic Benchmark",
        "=" * 40,
        f"Cases:      {metrics['total_cases']}",
        f"Evaluated:  {metrics['evaluated']}",
        f"Missing AI: {metrics['missing_analysis']}",
        f"Matches:    {metrics['matches']}/{metrics['evaluated']}",
        f"Accuracy:   {metrics['accuracy_pct']}%",
        "",
        "By expected label:",
    ]
    for label, counts in sorted(metrics["by_expected"].items()):
        pct = round(counts["correct"] / counts["total"] * 100, 1) if counts["total"] else 0
        lines.append(f"  {label:8} {counts['correct']}/{counts['total']} ({pct}%)")

    lines.extend(["", "Details:"])
    for row in results:
        ai = row.get("ai_decision") or "N/A"
        exp = row.get("expected_decision") or "N/A"
        mark = "OK" if row.get("match") else "X"
        conf = row.get("confidence_score")
        conf_str = f" {conf:.0%}" if conf is not None else ""
        lines.append(
            f"  [{mark:2}] [{row.get('sample_index', '?'):>2}] "
            f"{row.get('patient_label', '?'):4} {row.get('procedure_label', '')[:35]:35} "
            f"expected={exp:8} ai={ai:8}{conf_str}"
        )
    return "\n".join(lines)


def run_eval(db: Session | None = None) -> dict:
    """Run evaluation and return metrics + row-level results."""
    own_session = db is None
    if own_session:
        init_db()
        db = SessionLocal()

    try:
        results = load_benchmark_rows(db)
        if not results:
            results = load_expected_from_json()

        metrics = compute_metrics(results)
        return {"metrics": metrics, "results": results}
    finally:
        if own_session and db is not None:
            db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AI vs expected synthetic decisions")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text report")
    args = parser.parse_args()

    output = run_eval()
    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(format_report(output["results"], output["metrics"]))

    metrics = output["metrics"]
    if metrics["evaluated"] == 0:
        print("\nNo analyzed synthetic cases found. Start the server and wait for pre-analysis, then re-run.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
