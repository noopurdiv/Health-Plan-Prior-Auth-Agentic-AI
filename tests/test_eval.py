from eval import compute_metrics, decisions_match


def test_decisions_match_exact():
    assert decisions_match("APPROVE", "APPROVE")
    assert decisions_match("FLAG", "flag")
    assert not decisions_match("APPROVE", "ESCALATE")
    assert not decisions_match(None, "APPROVE")


def test_compute_metrics_accuracy():
    results = [
        {"expected_decision": "APPROVE", "ai_decision": "APPROVE", "match": True},
        {"expected_decision": "FLAG", "ai_decision": "ESCALATE", "match": False},
        {"expected_decision": "ESCALATE", "ai_decision": "ESCALATE", "match": True},
        {"expected_decision": "APPROVE", "ai_decision": None, "match": False},
    ]
    metrics = compute_metrics(results)
    assert metrics["total_cases"] == 4
    assert metrics["evaluated"] == 3
    assert metrics["missing_analysis"] == 1
    assert metrics["matches"] == 2
    assert metrics["accuracy_pct"] == 66.7
    assert metrics["by_expected"]["APPROVE"]["correct"] == 1
    assert metrics["by_expected"]["APPROVE"]["total"] == 1
