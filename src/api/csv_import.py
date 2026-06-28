import csv
import io
import uuid
from typing import Any

from sqlalchemy.orm import Session

from src.db.models import PriorAuthRequest

REQUIRED_COLUMNS = {"patient_age", "patient_sex", "procedure_label"}
COLUMN_ALIASES = {
    "age": "patient_age",
    "sex": "patient_sex",
    "gender": "patient_sex",
    "icd10": "diagnosis_code",
    "icd_10": "diagnosis_code",
    "diagnosis": "diagnosis_label",
    "procedure": "procedure_label",
    "cpt": "procedure_code",
    "notes": "clinical_notes",
}


def _normalize_header(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return COLUMN_ALIASES.get(key, key)


def parse_csv_rows(content: bytes) -> list[dict[str, Any]]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV has no header row")

    normalized_fields = {_normalize_header(f): f for f in reader.fieldnames}
    missing = REQUIRED_COLUMNS - set(normalized_fields.keys())
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")

    rows: list[dict[str, Any]] = []
    for i, raw in enumerate(reader, start=2):
        row = {_normalize_header(k): (v or "").strip() for k, v in raw.items() if k}
        try:
            age = int(row.get("patient_age", 0))
            if age < 1 or age > 120:
                raise ValueError(f"row {i}: patient_age must be 1-120")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {i}: invalid patient_age") from exc

        sex = row.get("patient_sex", "").upper()
        if sex not in ("M", "F", "MALE", "FEMALE"):
            raise ValueError(f"row {i}: patient_sex must be M or F")
        sex = "M" if sex.startswith("M") else "F"

        procedure = row.get("procedure_label", "")
        if not procedure:
            raise ValueError(f"row {i}: procedure_label is required")

        urgency = (row.get("urgency") or "routine").lower()
        if urgency not in ("routine", "urgent", "emergent"):
            urgency = "routine"

        rows.append(
            {
                "patient_age": age,
                "patient_sex": sex,
                "diagnosis_code": row.get("diagnosis_code") or "N/A",
                "diagnosis_label": row.get("diagnosis_label") or "Unspecified",
                "procedure_code": row.get("procedure_code") or "",
                "procedure_label": procedure,
                "clinical_notes": row.get("clinical_notes") or "",
                "urgency": urgency,
                "expected_decision": row.get("expected_decision") or None,
            }
        )
    return rows


def import_csv_requests(db: Session, content: bytes) -> list[PriorAuthRequest]:
    rows = parse_csv_rows(content)
    created: list[PriorAuthRequest] = []
    for row in rows:
        req = PriorAuthRequest(
            id=str(uuid.uuid4()),
            patient_age=row["patient_age"],
            patient_sex=row["patient_sex"],
            diagnosis_code=row["diagnosis_code"],
            diagnosis_label=row["diagnosis_label"],
            procedure_code=row["procedure_code"],
            procedure_label=row["procedure_label"],
            clinical_notes=row["clinical_notes"],
            urgency=row["urgency"],
            status="pending",
            is_synthetic=False,
            expected_decision=row.get("expected_decision"),
        )
        db.add(req)
        created.append(req)
    db.commit()
    return created
