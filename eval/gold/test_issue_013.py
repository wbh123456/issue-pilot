"""Hidden gold test for issue-013 — evaluator only."""

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


def test_receipt_follows_updated_address_without_relogin():
    headers = _login("alice@example.com", "alicepass1")
    updated = client.patch(
        "/users/me/email",
        json={"email": "updated@example.com"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["email"] == "updated@example.com"

    profile = client.get("/users/2")
    assert profile.status_code == 200
    assert profile.json()["email"] == "updated@example.com"

    created = client.post(
        "/orders",
        json={"items": [{"sku": "gadget", "price": 5.0, "qty": 1}]},
        headers=headers,
    )
    assert created.status_code == 200

    inbox = client.get("/notifications/inbox", headers=headers)
    assert inbox.status_code == 200
    messages = inbox.json()["messages"]
    assert messages
    assert messages[-1]["to"] == "updated@example.com"
    assert "alice@example.com" not in {item["to"] for item in messages}
