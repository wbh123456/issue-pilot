"""Hidden gold test for issue-011 — evaluator only."""

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


def test_coupon_applied_once_and_refund_restores_stock():
    token = _login_alice()
    headers = {"Authorization": f"Bearer {token}"}
    before = inventory.get_stock("widget")

    created = client.post(
        "/orders",
        json={
            "items": [{"sku": "widget", "price": 10.0, "qty": 1}],
            "coupon": "SAVE10",
        },
        headers=headers,
    )
    assert created.status_code == 200
    body = created.json()
    assert body["total"] == 9.0
    assert inventory.get_stock("widget") == before - 1

    twenty = client.post(
        "/orders",
        json={
            "items": [{"sku": "gadget", "price": 10.0, "qty": 1}],
            "coupon": "SAVE20",
        },
        headers=headers,
    )
    assert twenty.status_code == 200
    assert twenty.json()["total"] == 8.0

    refunded = client.post(
        f"/orders/{body['id']}/refund",
        headers=headers,
    )
    assert refunded.status_code == 200
    assert inventory.get_stock("widget") == before

    again = client.post(
        f"/orders/{body['id']}/refund",
        headers=headers,
    )
    assert again.status_code in {200, 400, 409, 422}
    assert inventory.get_stock("widget") == before
