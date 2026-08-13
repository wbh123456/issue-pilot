"""Hidden gold test for issue-005 — evaluator only."""

from app import orders


def test_calculate_total_respects_quantity():
    total = orders.calculate_total(
        [
            {"price": 10.0, "qty": 3},
            {"price": 2.5, "qty": 2},
        ]
    )
    assert total == 35.0
