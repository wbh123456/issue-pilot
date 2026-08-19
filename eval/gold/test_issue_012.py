"""Hidden gold test for issue-012 — evaluator only."""

from fastapi.testclient import TestClient

from app import inventory, orders, users
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    users.reset_store()
    orders.reset_store()
    inventory.reset_store()


def _login_alice() -> dict[str, str]:
    r = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "alicepass1"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_cancel_appends_cancellation_for_the_same_actor():
    headers = _login_alice()
    created = client.post(
        "/orders",
        json={"items": [{"sku": "gadget", "price": 5.0, "qty": 1}]},
        headers=headers,
    )
    assert created.status_code == 200
    order_id = created.json()["id"]

    before = client.get(f"/audit/events?order_id={order_id}", headers=headers)
    assert before.status_code == 200
    opened = before.json()["events"]
    assert opened
    assert opened[-1]["kind"] == "sale"
    assert opened[-1]["actor_id"] == 2

    cancelled = client.post(f"/orders/{order_id}/refund", headers=headers)
    assert cancelled.status_code == 200

    after = client.get(f"/audit/events?order_id={order_id}", headers=headers)
    assert after.status_code == 200
    events = after.json()["events"]
    assert events[-1]["kind"] == "cancellation"
    assert events[-1]["actor_id"] == 2
    assert any(item["kind"] == "sale" for item in events)
    assert any(item["kind"] == "cancellation" for item in events)
