"""Hidden gold test for issue-010 — evaluator only."""

from fastapi.testclient import TestClient

from app import users
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    users.reset_store()


def test_promote_takes_effect_without_relogin():
    alice_login = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "alicepass1"},
    )
    assert alice_login.status_code == 200
    alice_token = alice_login.json()["access_token"]

    bob_login = client.post(
        "/auth/login",
        json={"email": "bob@example.com", "password": "bobpass1"},
    )
    assert bob_login.status_code == 200
    bob_token = bob_login.json()["access_token"]

    users.promote_user(2)

    r = client.post(
        "/users/2/promote",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "admin"

    denied = client.post(
        "/users/1/promote",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert denied.status_code == 403
