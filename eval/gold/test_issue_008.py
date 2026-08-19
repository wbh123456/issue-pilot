"""Hidden gold test for issue-008 — evaluator only."""

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


def test_idempotency_is_scoped_to_user():
    alice = _login("alice@example.com", "alicepass1")
    bob = _login("bob@example.com", "bobpass1")
    key = "abc-123"
    body = {"items": [{"sku": "gadget", "price": 5.0, "qty": 1}]}
    other = {"items": [{"sku": "widget", "price": 5.0, "qty": 1}]}

    r1 = client.post(
        "/orders", json=body, headers={**alice, "Idempotency-Key": key}
    )
    r2 = client.post(
        "/orders", json=body, headers={**alice, "Idempotency-Key": key}
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]
    assert r1.json()["user_id"] == 2

    mismatch = client.post(
        "/orders", json=other, headers={**alice, "Idempotency-Key": key}
    )
    assert mismatch.status_code in {400, 409, 422}
    replay = client.post(
        "/orders", json=body, headers={**alice, "Idempotency-Key": key}
    )
    assert replay.status_code == 200
    assert replay.json()["id"] == r1.json()["id"]

    r3 = client.post(
        "/orders", json=body, headers={**bob, "Idempotency-Key": key}
    )
    assert r3.status_code == 200
    assert r3.json()["id"] != r1.json()["id"]
    assert r3.json()["user_id"] == 3

    r4 = client.post("/orders", json=body, headers=alice)
    r5 = client.post("/orders", json=body, headers=alice)
    assert r4.status_code == 200 and r5.status_code == 200
    assert r4.json()["id"] != r5.json()["id"]
