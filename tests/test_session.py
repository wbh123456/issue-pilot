"""RunSession sidecar round-trip (runs/sessions/, not eval report input)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from eval.session import (
    SESSIONS_DIR,
    RunSession,
    list_sessions,
    load_session,
    new_run_id,
    save_session,
    session_from_dict,
    session_path,
    update_session,
)


def _session(**overrides: str) -> RunSession:
    payload = {
        "run_id": "issue-001-v1-20260817T000000Z",
        "thread_id": "issue-001-v1-20260817T000000Z",
        "task_id": "issue-001",
        "harness": "v1",
        "model": "deepseek-v4-flash",
        "embedder_name": "hashing",
        "query_mode": "issue",
        "repo_path": "C:/fake/benchmark",
        "base_commit": "abc123",
        "status": "running",
        "created_at": "2026-08-17T00:00:00Z",
    }
    payload.update(overrides)
    return RunSession(**payload)


class TestRunSession:
    def test_default_dir_is_runs_sessions(self) -> None:
        assert SESSIONS_DIR.name == "sessions"
        assert SESSIONS_DIR.parent.name == "runs"
        path = session_path("issue-001-v1-demo")
        assert path.parent == SESSIONS_DIR
        assert path.name == "issue-001-v1-demo.json"

    def test_round_trip(self, tmp_path: Path) -> None:
        session = _session()
        path = save_session(session, sessions_dir=tmp_path)
        assert path == tmp_path / f"{session.run_id}.json"
        loaded = load_session(session.run_id, sessions_dir=tmp_path)
        assert loaded == session
        assert json.loads(path.read_text(encoding="utf-8")) == asdict(session)
        assert loaded.resume_count == 0

    def test_resume_count_round_trip_and_update(self, tmp_path: Path) -> None:
        original = _session()
        save_session(original, sessions_dir=tmp_path)
        loaded = load_session(original.run_id, sessions_dir=tmp_path)
        assert loaded.resume_count == 0
        updated = update_session(
            loaded, sessions_dir=tmp_path, status="paused", resume_count=2
        )
        assert updated.status == "paused"
        assert updated.resume_count == 2
        assert load_session(updated.run_id, sessions_dir=tmp_path).resume_count == 2

    def test_resume_count_defaults_when_missing(self) -> None:
        payload = {
            "run_id": "issue-001-v1-20260817T000000Z",
            "thread_id": "issue-001-v1-20260817T000000Z",
            "task_id": "issue-001",
            "harness": "v1",
            "model": "deepseek-v4-flash",
            "embedder_name": "hashing",
            "query_mode": "issue",
            "repo_path": "C:/fake/benchmark",
            "base_commit": "abc123",
            "status": "paused",
            "created_at": "2026-08-17T00:00:00Z",
        }
        loaded = session_from_dict(payload)
        assert loaded.resume_count == 0

    def test_list_skips_corrupt_and_sorts(self, tmp_path: Path) -> None:
        first = save_session(
            _session(run_id="a-run", thread_id="a-run", status="paused"),
            sessions_dir=tmp_path,
        )
        save_session(
            _session(run_id="b-run", thread_id="b-run", status="running"),
            sessions_dir=tmp_path,
        )
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "incomplete.json").write_text(
            json.dumps({"run_id": "x"}),
            encoding="utf-8",
        )
        listed = list_sessions(sessions_dir=tmp_path)
        assert [item.run_id for item in listed] == ["a-run", "b-run"]
        assert listed[0].status == "paused"
        assert first.parent == tmp_path

    def test_load_missing_and_invalid_id(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_session("missing-run", sessions_dir=tmp_path)
        with pytest.raises(ValueError, match="invalid run_id"):
            session_path("../escape", sessions_dir=tmp_path)
        with pytest.raises(ValueError, match="missing fields"):
            session_from_dict({"run_id": "x"})

    def test_new_run_id_uses_task_and_harness(self) -> None:
        run_id = new_run_id("issue-009", "v2")
        assert run_id.startswith("issue-009-v2-")
        assert run_id.endswith("Z")
