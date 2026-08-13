"""Hidden gold test for issue-004 — evaluator only."""

from fastapi.testclient import TestClient

from app import users
from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def setup_function(_):
    users.reset_store()


def test_missing_user_returns_404():
    r = client.get("/users/99999")
    assert r.status_code == 404
