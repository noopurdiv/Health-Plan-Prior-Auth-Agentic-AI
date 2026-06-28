import json
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from src.db.models import PriorAuthRequest

ROOT_DIR = Path(__file__).resolve().parents[2]
SAMPLE_PATH = ROOT_DIR / "synthetic_data" / "sample_requests.json"


def seed_synthetic_requests(db: Session) -> int:
    """Load sample requests from JSON; add any sample_index not yet in the database."""
    if not SAMPLE_PATH.exists():
        return 0

    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    existing_indices = {
        row[0]
        for row in db.query(PriorAuthRequest.sample_index)
        .filter(PriorAuthRequest.is_synthetic == True)  # noqa: E712
        .all()
        if row[0] is not None
    }

    added = 0
    for i, sample in enumerate(samples):
        if i in existing_indices:
            continue
        db.add(
            PriorAuthRequest(
                id=str(uuid.uuid4()),
                patient_age=sample["patient_age"],
                patient_sex=sample["patient_sex"],
                diagnosis_code=sample["diagnosis_code"],
                diagnosis_label=sample["diagnosis_label"],
                procedure_code=sample.get("procedure_code", ""),
                procedure_label=sample["procedure_label"],
                clinical_notes=sample.get("clinical_notes", ""),
                urgency=sample.get("urgency", "routine").lower(),
                status="pending",
                is_synthetic=True,
                expected_decision=sample.get("expected_decision"),
                sample_index=i,
            )
        )
        added += 1

    if added:
        db.commit()
    return added
