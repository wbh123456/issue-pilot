"""Hidden gold test for issue-006 — evaluator only."""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi.testclient import TestClient

from app import auth, users
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    users.reset_store()


def _token_without_user_id() -> str:
    payload = {"exp": datetime.now(timezone.utc) + timedelta(seconds=60)}
    return jwt.encode(payload, auth.SECRET, algorithm=auth.ALGO)


def test_token_without_user_id_returns_401():
    token = _token_without_user_id()
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
