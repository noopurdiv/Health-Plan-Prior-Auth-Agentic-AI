from src.api.main import _map_final_status, _map_request_status


def test_map_final_status_approve():
    assert _map_final_status("approve") == "approved"
    assert _map_final_status("APPROVE") == "approved"


def test_map_final_status_reject_and_escalate():
    assert _map_final_status("reject") == "rejected"
    assert _map_final_status("escalate") == "escalated"


def test_map_request_status_matches_final_status():
    for action in ("approve", "reject", "escalate", "override"):
        assert _map_request_status(action) == _map_final_status(action)
