"""Hidden gold test for issue-002 — evaluator only."""

from app.calculator import sum_inclusive


def test_sum_inclusive_basic():
    assert sum_inclusive(1, 5) == 15
    assert sum_inclusive(3, 3) == 3
