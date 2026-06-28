import pytest

from src.api.csv_import import parse_csv_rows

VALID_CSV = b"""patient_age,patient_sex,procedure_label,diagnosis_code,urgency
58,F,Colonoscopy,Z12.11,routine
"""


def test_parse_csv_rows_valid():
    rows = parse_csv_rows(VALID_CSV)
    assert len(rows) == 1
    assert rows[0]["patient_age"] == 58
    assert rows[0]["patient_sex"] == "F"
    assert rows[0]["procedure_label"] == "Colonoscopy"
    assert rows[0]["urgency"] == "routine"


def test_parse_csv_rows_missing_required_column():
    bad_csv = b"patient_age,procedure_label\n58,Colonoscopy\n"
    with pytest.raises(ValueError, match="missing required columns"):
        parse_csv_rows(bad_csv)


def test_parse_csv_rows_invalid_age():
    bad_csv = b"patient_age,patient_sex,procedure_label\n0,F,Colonoscopy\n"
    with pytest.raises(ValueError, match="patient_age"):
        parse_csv_rows(bad_csv)
