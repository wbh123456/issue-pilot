"""Hidden gold test for issue-001 — evaluator only."""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi.testclient import TestClient

from app import auth, users
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    users.reset_store()


def _expired_token(user_id: int = 1) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) - timedelta(seconds=30),
    }
    return jwt.encode(payload, auth.SECRET, algorithm=auth.ALGO)


def test_expired_token_returns_401():
    token = _expired_token()
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401
