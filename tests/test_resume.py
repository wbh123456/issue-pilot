"""Pause / resume runner: durable interrupt, no reset, gold stays independent."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import eval.runner as runner
from eval.runner import load_review_payload, resume_task, solve_task
from eval.session import load_session
from tests.test_checkpoint import _pass_patches
from tests.test_graph import FakeClient, VALID_EVALUATION, VALID_PLAN, _Response
from tests.test_runner import FakeSandbox, _agent_result, _fake_task


@pytest.fixture
def fake_sandbox(monkeypatch: pytest.MonkeyPatch) -> type[FakeSandbox]:
    FakeSandbox.reset()
    monkeypatch.setattr(runner, "SandboxRunner", FakeSandbox)
    return FakeSandbox


def _gold_ok() -> dict[str, Any]:
    return {
        "command": "pytest gold",
        "exit_code": 0,
        "passed": True,
        "output": "exit_code=0",
    }


def _gold_fail() -> dict[str, Any]:
    return {
        "command": "pytest gold",
        "exit_code": 1,
        "passed": False,
        "output": "exit_code=1",
    }


def _waiting_result() -> dict[str, Any]:
    result = _agent_result("v1")
    result["status"] = "waiting_approval"
    result["termination"] = "waiting_approval"
    result["workflow_passed"] = False
    result["approval_payload"] = {
        "issue": "Expired JWT returns 500 instead of 401",
        "plan": result["plan"],
        "changed_files": ["app/auth.py"],
        "git_diff": "diff --git a/app/auth.py",
        "test_result": result["test_result"],
        "evaluator_result": {},
    }
    return result


def _pass_client() -> FakeClient:
    return FakeClient(
        [
            _Response("Problem: expired JWT. Hypothesis: missing catch."),
            _Response(json.dumps(VALID_PLAN)),
            _Response(json.dumps(VALID_EVALUATION)),
        ]
    )


class TestPauseResumeControlFlow:
    def test_v0_rejects_require_approval(self) -> None:
        with pytest.raises(ValueError, match="v1 or v2"):
            solve_task("issue-001", harness_version="v0", require_approval=True)

    def test_pause_writes_session_not_run_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        repo = tmp_path / "bench"
        repo.mkdir()
        task = _fake_task()
        sessions = tmp_path / "sessions"
        ckpt = tmp_path / "ck.sqlite"
        gold_mock = patch.object(runner, "run_gold_test")

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=repo),
            patch.object(runner, "reset_repo") as reset_mock,
            patch.object(runner, "_run_harness", return_value=_waiting_result()),
            gold_mock as gold,
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            paused = solve_task(
                "issue-001",
                harness_version="v1",
                require_approval=True,
                checkpoint_path=ckpt,
                sessions_dir=sessions,
            )

        reset_mock.assert_called_once()
        gold.assert_not_called()
        assert paused["paused"] is True
        assert paused["status"] == "waiting_approval"
        assert "run_path" not in paused
        assert list(tmp_path.glob("*.json")) == []
        session = load_session(paused["run_id"], sessions_dir=sessions)
        assert session.status == "paused"
        assert session.resume_count == 0
        assert session.repo_path == str(repo)

    def test_resume_finishes_without_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        repo = tmp_path / "bench"
        repo.mkdir()
        task = _fake_task()
        sessions = tmp_path / "sessions"
        ckpt = tmp_path / "ck.sqlite"
        approved = _agent_result("v1")
        approved["approval_decision"] = "approve"
        approved["status"] = "success"

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=repo),
            patch.object(runner, "reset_repo") as reset_mock,
            patch.object(runner, "_run_harness", return_value=_waiting_result()),
            patch.object(runner, "run_gold_test", return_value=_gold_ok()) as gold,
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
            patch.object(runner, "git_sha", return_value="deadbeef"),
            patch.object(runner, "verify_resume_worktree") as verify,
            patch.object(runner, "resume_workflow", return_value=approved),
        ):
            paused = solve_task(
                "issue-001",
                harness_version="v1",
                require_approval=True,
                checkpoint_path=ckpt,
                sessions_dir=sessions,
            )
            record = resume_task(
                paused["run_id"],
                decision="approve",
                checkpoint_path=ckpt,
                sessions_dir=sessions,
            )

        reset_mock.assert_called_once()
        verify.assert_called_once()
        gold.assert_called_once()
        assert record["resumed"] is True
        assert record["resume_count"] == 1
        assert record["sandbox_sessions"] == 2
        assert record["approval_decision"] == "approve"
        assert record["success"] is True
        assert record["run_id"] == paused["run_id"]
        assert Path(record["run_path"]).name == f"{paused['run_id']}.json"
        assert Path(record["run_path"]).is_file()
        assert len(fake_sandbox.instances) == 2
        assert load_session(paused["run_id"], sessions_dir=sessions).status == "completed"

    def test_resume_gold_independent_of_approval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        repo = tmp_path / "bench"
        repo.mkdir()
        task = _fake_task()
        sessions = tmp_path / "sessions"
        ckpt = tmp_path / "ck.sqlite"
        approved = _agent_result("v1")
        approved["approval_decision"] = "approve"
        approved["workflow_passed"] = True
        approved["status"] = "success"

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=repo),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", return_value=_waiting_result()),
            patch.object(runner, "run_gold_test", return_value=_gold_fail()),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
            patch.object(runner, "git_sha", return_value="deadbeef"),
            patch.object(runner, "verify_resume_worktree"),
            patch.object(runner, "resume_workflow", return_value=approved),
        ):
            paused = solve_task(
                "issue-001",
                harness_version="v1",
                require_approval=True,
                checkpoint_path=ckpt,
                sessions_dir=sessions,
            )
            record = resume_task(
                paused["run_id"],
                decision="approve",
                checkpoint_path=ckpt,
                sessions_dir=sessions,
            )

        assert record["approval_decision"] == "approve"
        assert record["workflow_passed"] is True
        assert record["success"] is False

    def test_resume_rejects_non_paused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        repo = tmp_path / "bench"
        repo.mkdir()
        task = _fake_task()
        sessions = tmp_path / "sessions"
        ckpt = tmp_path / "ck.sqlite"
        approved = _agent_result("v1")
        approved["approval_decision"] = "approve"
        approved["status"] = "success"

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=repo),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", return_value=_waiting_result()),
            patch.object(runner, "run_gold_test", return_value=_gold_ok()),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
            patch.object(runner, "git_sha", return_value="deadbeef"),
            patch.object(runner, "verify_resume_worktree"),
            patch.object(runner, "resume_workflow", return_value=approved),
        ):
            paused = solve_task(
                "issue-001",
                harness_version="v1",
                require_approval=True,
                checkpoint_path=ckpt,
                sessions_dir=sessions,
            )
            resume_task(
                paused["run_id"],
                decision="approve",
                checkpoint_path=ckpt,
                sessions_dir=sessions,
            )
            with pytest.raises(ValueError, match="not paused"):
                resume_task(
                    paused["run_id"],
                    decision="approve",
                    checkpoint_path=ckpt,
                    sessions_dir=sessions,
                )


class TestPauseResumeFreshSaver:
    def test_pause_then_finish_via_fresh_saver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        repo = tmp_path / "bench"
        repo.mkdir()
        task = _fake_task()
        sessions = tmp_path / "sessions"
        ckpt = tmp_path / "ck.sqlite"
        client = _pass_client()

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "get_task", return_value=task))
            stack.enter_context(patch.object(runner, "resolve_repo_path", return_value=repo))
            reset_mock = stack.enter_context(patch.object(runner, "reset_repo"))
            gold = stack.enter_context(
                patch.object(runner, "run_gold_test", return_value=_gold_ok())
            )
            stack.enter_context(patch.object(runner, "git_sha", return_value="deadbeef"))
            stack.enter_context(patch.object(runner, "verify_resume_worktree"))
            for patcher in _pass_patches():
                stack.enter_context(patcher)
            paused = solve_task(
                "issue-001",
                harness_version="v1",
                require_approval=True,
                client=client,
                checkpoint_path=ckpt,
                sessions_dir=sessions,
                max_steps=5,
            )

        assert paused["paused"] is True
        assert "run_path" not in paused
        assert list(tmp_path.glob("*.json")) == []
        gold.assert_not_called()
        reset_mock.assert_called_once()
        payload = load_review_payload(
            paused["run_id"],
            checkpoint_path=ckpt,
            sessions_dir=sessions,
        )
        assert set(payload) >= {
            "issue",
            "plan",
            "changed_files",
            "git_diff",
            "test_result",
            "evaluator_result",
        }
        assert "Expired JWT" in payload["issue"]
        assert payload["plan"]["problem"] == VALID_PLAN["problem"]

        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "get_task", return_value=task))
            stack.enter_context(patch.object(runner, "git_sha", return_value="deadbeef"))
            stack.enter_context(patch.object(runner, "verify_resume_worktree"))
            gold = stack.enter_context(
                patch.object(runner, "run_gold_test", return_value=_gold_ok())
            )
            for patcher in _pass_patches():
                stack.enter_context(patcher)
            record = resume_task(
                paused["run_id"],
                decision="approve",
                client=FakeClient([]),
                checkpoint_path=ckpt,
                sessions_dir=sessions,
                max_steps=5,
            )

        reset_mock.assert_called_once()
        gold.assert_called_once()
        assert record["success"] is True
        assert record["resumed"] is True
        assert record["resume_count"] == 1
        assert record["sandbox_sessions"] == 2
        assert record["approval_decision"] == "approve"
        assert record["status"] == "success"
        assert record["run_id"] == paused["run_id"]
        assert Path(record["run_path"]).is_file()
        assert len(fake_sandbox.instances) == 2
        assert load_session(paused["run_id"], sessions_dir=sessions).status == "completed"
