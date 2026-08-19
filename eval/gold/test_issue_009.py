"""Hidden gold test for issue-009 — evaluator only."""

from fastapi.testclient import TestClient

from app import inventory, orders, users
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    users.reset_store()
    orders.reset_store()
    inventory.reset_store()


def _login_alice() -> str:
    r = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "alicepass1"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_out_of_stock_returns_409_and_does_not_decrement():
    token = _login_alice()
    before = inventory.get_stock("widget")
    assert before == 3

    r = client.post(
        "/orders",
        json={"items": [{"sku": "widget", "price": 9.0, "qty": 50}]},
        headers=_auth(token),
    )
    assert r.status_code == 409
    assert inventory.get_stock("widget") == before

    partial = client.post(
        "/orders",
        json={"items": [{"sku": "widget", "price": 9.0, "qty": 2}]},
        headers=_auth(token),
    )
    assert partial.status_code == 200
    assert inventory.get_stock("widget") == before - 2

    short = client.post(
        "/orders",
        json={"items": [{"sku": "widget", "price": 9.0, "qty": 2}]},
        headers=_auth(token),
    )
    assert short.status_code == 409
    assert inventory.get_stock("widget") == before - 2

    orders.reset_store()
    inventory.reset_store()
    gadget_before = inventory.get_stock("gadget")
    widget_before = inventory.get_stock("widget")
    mixed = client.post(
        "/orders",
        json={
            "items": [
                {"sku": "gadget", "price": 5.0, "qty": 1},
                {"sku": "widget", "price": 9.0, "qty": 50},
            ]
        },
        headers=_auth(token),
    )
    assert mixed.status_code == 409
    assert inventory.get_stock("gadget") == gadget_before
    assert inventory.get_stock("widget") == widget_before

    ok = client.post(
        "/orders",
        json={"items": [{"sku": "widget", "price": 9.0, "qty": 1}]},
        headers=_auth(token),
    )
    assert ok.status_code == 200
    assert inventory.get_stock("widget") == widget_before - 1
