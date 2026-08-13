"""Hidden gold test for issue-003 — evaluator only."""

from app.validators import is_valid_email


def test_empty_email_rejected():
    assert is_valid_email("") is False
    assert is_valid_email("   ") is False
