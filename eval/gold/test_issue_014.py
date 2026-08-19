"""Hidden gold test for issue-014 — evaluator only."""

from fastapi.testclient import TestClient

from app import inventory, orders, users
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    users.reset_store()
    orders.reset_store()
    inventory.reset_store()


def _login(email: str, password: str) -> dict[str, str]:
    r = client.post("/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_sales_summary_counts_units_and_skips_refunded():
    alice = _login("alice@example.com", "alicepass1")
    admin = _login("admin@example.com", "adminpass1")

    created = client.post(
        "/orders",
        json={"items": [{"sku": "gadget", "price": 10.0, "qty": 2}]},
        headers=alice,
    )
    assert created.status_code == 200
    order_id = created.json()["id"]

    summary = client.get("/reports/sales", headers=admin)
    assert summary.status_code == 200
    assert summary.json()["merchandise"] == 20.0

    denied = client.get("/reports/sales", headers=alice)
    assert denied.status_code == 403

    cancelled = client.post(f"/orders/{order_id}/refund", headers=alice)
    assert cancelled.status_code == 200

    after_cancel = client.get("/reports/sales", headers=admin)
    assert after_cancel.status_code == 200
    assert after_cancel.json()["merchandise"] == 0.0

    leftover = client.post(
        "/orders",
        json={"items": [{"sku": "gadget", "price": 5.0, "qty": 1}]},
        headers=alice,
    )
    assert leftover.status_code == 200

    rest = client.get("/reports/sales", headers=admin)
    assert rest.status_code == 200
    assert rest.json()["merchandise"] == 5.0
