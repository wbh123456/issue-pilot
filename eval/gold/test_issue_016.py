"""Hidden gold test for issue-016 — evaluator only."""

from fastapi.testclient import TestClient

from app import settings
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def _login_alice() -> dict[str, str]:
    r = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "alicepass1"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_late_checkout_confirmation_is_delivered():
    settings.enable_flag("quiet_hours")
    headers = _login_alice()
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
    assert messages[-1]["template"] == "receipt"
    assert messages[-1].get("skipped") is not True
