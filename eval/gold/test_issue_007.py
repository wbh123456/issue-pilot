"""Hidden gold test for issue-007 — evaluator only."""

import jwt
from fastapi.testclient import TestClient

from app import auth, users
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    users.reset_store()


def test_admin_token_carries_role_claim():
    login = client.post(
        "/auth/login",
        json={"email": "admin@example.com", "password": "adminpass1"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    payload = jwt.decode(token, auth.SECRET, algorithms=[auth.ALGO])
    assert payload.get("role") == "admin"

    r = client.post(
        "/users/2/promote",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["role"] == "admin"
