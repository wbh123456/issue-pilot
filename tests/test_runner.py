"""Mocked tests for versioned runner dispatch and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import cli
import eval.runner as runner
from eval.runner import _normalize_harness, _run_harness, save_run, solve_task
from sandbox.runner import SandboxError, SandboxMetadata


class FakeSandbox:
    """Context-managed stand-in so solve_task tests never talk to Docker."""

    instances: list[FakeSandbox] = []

    def __init__(self, workspace_host, *, task_id=None, **kwargs):
        self.workspace_host = workspace_host
        self.task_id = task_id
        self.meta = SandboxMetadata(
            image="issue-pilot-sandbox:py312",
            task_id=task_id,
            workspace_host=str(workspace_host),
        )
        type(self).instances.append(self)

    def __enter__(self) -> FakeSandbox:
        self.meta.started = True
        self.meta.usable = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.meta.cleaned_up = True
        self.meta.usable = False
        return None

    @classmethod
    def reset(cls) -> None:
        cls.instances = []


class FailingStartSandbox(FakeSandbox):
    def __enter__(self) -> FakeSandbox:
        raise SandboxError("docker daemon not reachable")


@pytest.fixture
def fake_sandbox(monkeypatch: pytest.MonkeyPatch) -> type[FakeSandbox]:
    FakeSandbox.reset()
    monkeypatch.setattr(runner, "SandboxRunner", FakeSandbox)
    return FakeSandbox


def _fake_task() -> dict[str, Any]:
    return {
        "id": "issue-001",
        "difficulty": "easy",
        "issue": "Expired JWT returns 500 instead of 401",
        "base_commit": "abc123",
        "repo_path": "C:/fake/benchmark",
        "split": "smoke",
        "test_command": "pytest tests/test_auth_expired.py -q",
        "lint_command": "ruff check app",
        "gold_file": "test_issue_001.py",
        "gold_test": "test_expired_token_returns_401",
        "expected_files": ["app/auth.py"],
    }


def _agent_result(harness: str) -> dict[str, Any]:
    base = {
        "final_answer": f"done via {harness}",
        "termination": "completed",
        "steps": 4,
        "tool_call_count": 6,
        "file_reads": 2,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "tokens": 120,
        "latency": 1.5,
        "trajectory": [],
        "messages": [],
        "llm_calls": 4 if harness == "v0" else 7,
    }
    if harness in {"v1", "v2"}:
        base.update(
            {
                "analysis": "analysis",
                "plan": {"problem": "p", "hypothesis": "h", "files_to_inspect": [], "steps": ["a"]},
                "diagnosis": "",
                "test_result": {"exit_code": 0, "passed": True},
                "status": "success",
                "workflow_passed": True,
                "retry_count": 0,
                "human_retry_count": 0,
                "human_feedback": "",
                "attempt_history": [],
                "structured_diagnosis": {},
                "patch_evaluation": {},
            }
        )
    if harness == "v2":
        base.update(
            {
                "relevant_files": ["app/auth.py"],
                "retrieval_calls": 1,
                "retrieve_query": "Expired JWT returns 500 instead of 401",
            }
        )
    return base


class TestHarnessNormalization:
    def test_accepts_v0_v1_v2(self) -> None:
        assert _normalize_harness("v0") == "v0"
        assert _normalize_harness("V1") == "v1"
        assert _normalize_harness("v2") == "v2"

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="harness_version"):
            _normalize_harness("v3")


class TestRunHarness:
    def test_v0_uses_run_agent_without_search_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_agent(**kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return _agent_result("v0")

        monkeypatch.setattr(runner, "run_agent", fake_agent)
        monkeypatch.setattr(
            runner,
            "run_workflow",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("v1/v2")),
        )
        out = _run_harness(
            harness="v0",
            client=object(),
            issue="i",
            repo_path="p",
            test_command="pytest -q",
            model="m",
            max_steps=5,
        )
        assert "tools" not in seen
        assert "search_code_enabled" not in seen
        assert out["final_answer"] == "done via v0"

    def test_v1_does_not_enable_search_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_workflow(**kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return _agent_result("v1")

        monkeypatch.setattr(runner, "run_workflow", fake_workflow)
        monkeypatch.setattr(
            runner,
            "get_v2_graph",
            lambda: (_ for _ in ()).throw(AssertionError("v2 graph")),
        )
        out = _run_harness(
            harness="v1",
            client=object(),
            issue="i",
            repo_path="p",
            test_command="pytest -q",
            model="m",
            max_steps=5,
        )
        assert "graph" not in seen
        assert seen.get("enable_search_code") is None
        assert "relevant_files" not in out

    def test_v2_uses_v2_graph_and_enables_search_code(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        fake_graph = object()

        def fake_workflow(**kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return _agent_result("v2")

        monkeypatch.setattr(runner, "run_workflow", fake_workflow)
        monkeypatch.setattr(runner, "get_v2_graph", lambda: fake_graph)
        monkeypatch.setattr(
            runner,
            "run_agent",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("v0")),
        )
        out = _run_harness(
            harness="v2",
            client=object(),
            issue="i",
            repo_path="p",
            test_command="pytest -q",
            model="m",
            max_steps=5,
            sandbox=object(),
            embedder_name="fastembed",
            query_mode="issue+analysis",
        )
        assert seen["graph"] is fake_graph
        assert seen["enable_search_code"] is True
        assert seen["embedder_name"] == "fastembed"
        assert seen["query_mode"] == "issue+analysis"
        assert out["relevant_files"] == ["app/auth.py"]
        assert out["retrieval_calls"] == 1


class TestSaveRun:
    def test_filename_includes_harness(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        path = save_run({"task_id": "issue-001", "harness_version": "v1", "ok": True})
        assert path.name.startswith("issue-001-v1-")
        assert path.suffix == ".json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["harness_version"] == "v1"

    def test_filename_uses_run_id_when_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        path = save_run(
            {
                "task_id": "issue-001",
                "harness_version": "v1",
                "run_id": "issue-001-v1-pausedemo",
            }
        )
        assert path.name == "issue-001-v1-pausedemo.json"


class TestSolveTaskDispatch:
    def test_dispatches_v0_and_records_common_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()
        calls: list[str] = []

        def fake_run_harness(*, harness: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(harness)
            return _agent_result(harness)

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo") as reset_mock,
            patch.object(runner, "_run_harness", side_effect=fake_run_harness),
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest tests/test_auth.py::test_expired_token_returns_401 -q",
                    "exit_code": 0,
                    "passed": True,
                    "output": "exit_code=0",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
            patch.object(runner, "git_sha", return_value="deadbeef"),
        ):
            record = solve_task("issue-001", harness_version="v0", max_steps=9)

        assert calls == ["v0"]
        reset_mock.assert_called_once()
        assert record["harness_version"] == "v0"
        assert record["success"] is True
        assert record["temperature"] == 0
        assert record["harness_git_sha"] == "deadbeef"
        assert record["benchmark_spec_sha"] == runner.benchmark_spec_sha()
        assert record["patch_diff"] == ""
        assert record["llm_calls"] == 4
        assert record["tokens"] == 120
        assert "analysis" not in record
        assert "recovery_success" not in record
        assert "attempt_history" not in record
        assert "human_retry_count" not in record
        assert "resumed" not in record
        assert "run_id" not in record
        assert "sandbox_sessions" not in record
        assert Path(record["run_path"]).name.startswith("issue-001-v0-")
        assert record["sandbox_backend"] == "docker"
        assert record["sandbox_network"] == "none"
        assert record["sandbox_cleaned_up"] is True
        assert record["sandbox_started"] is True

    def test_dispatches_v1_and_records_workflow_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo") as reset_mock,
            patch.object(runner, "_run_harness", return_value=_agent_result("v1")) as harness_mock,
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest ...",
                    "exit_code": 0,
                    "passed": True,
                    "output": "exit_code=0",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            record = solve_task("issue-001", harness_version="v1", model="m", max_steps=11)

        reset_mock.assert_called_once()
        assert harness_mock.call_args.kwargs["harness"] == "v1"
        assert harness_mock.call_args.kwargs["max_steps"] == 11
        assert harness_mock.call_args.kwargs["sandbox"] is fake_sandbox.instances[0]
        assert harness_mock.call_args.kwargs["feedback_provider"] is None
        assert record["harness_version"] == "v1"
        assert record["analysis"] == "analysis"
        assert record["plan"]["steps"] == ["a"]
        assert record["workflow_passed"] is True
        assert record["verification"]["exit_code"] == 0
        assert record["retry_count"] == 0
        assert record["human_retry_count"] == 0
        assert record["recovery_success"] is False
        assert record["attempt_history"] == []
        assert record["patch_evaluation"] == {}
        assert record["structured_diagnosis"] == {}
        assert record["human_feedback"] == ""
        assert record["approval_decision"] == ""
        assert record["resumed"] is False
        assert record["resume_count"] == 0
        assert record["sandbox_sessions"] == 1
        assert record["retrieval_calls"] == 0
        assert record["workflow_trace"] == []
        assert record["checkpoint_stages"] == []
        assert "run_id" not in record
        assert "retrieval_mode" not in record
        assert "recall_at_5" not in record
        assert Path(record["run_path"]).name.startswith("issue-001-v1-")

    def test_records_patch_diff_from_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", return_value=_agent_result("v1")),
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest ...",
                    "exit_code": 0,
                    "passed": True,
                    "output": "exit_code=0",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
            patch.object(
                runner, "capture_patch_diff", return_value="diff --git a/app/auth.py\n"
            ),
        ):
            record = solve_task("issue-001", harness_version="v1")

        assert record["patch_diff"] == "diff --git a/app/auth.py\n"
        assert len(record["benchmark_spec_sha"]) == 64

    def test_records_recovery_success_only_after_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()
        recovered = _agent_result("v1")
        recovered["retry_count"] = 1
        recovered["workflow_passed"] = True
        recovered["attempt_history"] = [
            {"attempt_index": 0, "failure_source": "deterministic"}
        ]
        recovered["patch_evaluation"] = {
            "issue_resolved": True,
            "patch_scope": "appropriate",
            "regression_risk": "low",
            "missing_tests": False,
            "feedback": "",
        }
        recovered["structured_diagnosis"] = {"root_cause": "insufficient stock"}

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", return_value=recovered),
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest ...",
                    "exit_code": 0,
                    "passed": True,
                    "output": "exit_code=0",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            record = solve_task("issue-001", harness_version="v1")

        assert record["success"] is True
        assert record["retry_count"] == 1
        assert record["recovery_success"] is True
        assert record["attempt_history"][0]["failure_source"] == "deterministic"
        assert record["patch_evaluation"]["issue_resolved"] is True
        assert record["structured_diagnosis"]["root_cause"] == "insufficient stock"

    def test_interactive_recovery_installs_stdin_provider(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        from agent.nodes.feedback import stdin_feedback_provider

        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", return_value=_agent_result("v1")) as harness_mock,
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest ...",
                    "exit_code": 0,
                    "passed": True,
                    "output": "exit_code=0",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            solve_task("issue-001", harness_version="v1", interactive_recovery=True)

        assert harness_mock.call_args.kwargs["feedback_provider"] is stdin_feedback_provider

    def test_dispatches_v2_and_records_retrieval_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", return_value=_agent_result("v2")) as harness_mock,
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest ...",
                    "exit_code": 0,
                    "passed": True,
                    "output": "exit_code=0",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            record = solve_task(
                "issue-001",
                harness_version="v2",
                embedder_name="fastembed",
                query_mode="issue+analysis",
            )

        assert harness_mock.call_args.kwargs["harness"] == "v2"
        assert harness_mock.call_args.kwargs["embedder_name"] == "fastembed"
        assert harness_mock.call_args.kwargs["query_mode"] == "issue+analysis"
        assert record["harness_version"] == "v2"
        assert record["workflow_passed"] is True
        assert record["retrieval_mode"] == "hybrid"
        assert record["retrieval_calls"] == 1
        assert record["relevant_files"] == ["app/auth.py"]
        assert record["recall_at_5"] == 1.0
        assert record["embedder_name"] == "fastembed"
        assert record["query_mode"] == "issue+analysis"
        assert record["retrieve_query"] == "Expired JWT returns 500 instead of 401"
        assert record["retry_count"] == 0
        assert Path(record["run_path"]).name.startswith("issue-001-v2-")

    def test_gold_success_independent_of_workflow_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()
        v1 = _agent_result("v1")
        v1["workflow_passed"] = True
        v1["test_result"] = {"exit_code": 0, "passed": True}

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", return_value=v1),
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest gold",
                    "exit_code": 1,
                    "passed": False,
                    "output": "exit_code=1",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            record = solve_task("issue-001", harness_version="v1")

        assert record["workflow_passed"] is True
        assert record["success"] is False

    def test_compare_isolation_resets_each_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        """Each solve_task resets; compare relies on that for isolation."""
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()
        resets: list[str] = []

        def fake_reset(repo_path: Path, base_commit: str) -> None:
            resets.append(base_commit)

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo", side_effect=fake_reset),
            patch.object(
                runner,
                "_run_harness",
                side_effect=lambda **kw: _agent_result(kw["harness"]),
            ),
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest",
                    "exit_code": 0,
                    "passed": True,
                    "output": "exit_code=0",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            solve_task("issue-001", harness_version="v0")
            solve_task("issue-001", harness_version="v1")

        assert resets == ["abc123", "abc123"]
        assert len(fake_sandbox.instances) == 2
        assert fake_sandbox.instances[0] is not fake_sandbox.instances[1]
        assert all(sb.meta.cleaned_up for sb in fake_sandbox.instances)

    def test_one_container_shared_by_harness_and_gold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()
        seen: dict[str, Any] = {}

        def fake_run_harness(*, sandbox, **kwargs: Any) -> dict[str, Any]:
            seen["harness"] = sandbox
            sandbox.meta.command_count += 2
            return _agent_result("v0")

        def fake_gold(repo_path: Path, task: dict[str, Any], *, sandbox) -> dict[str, Any]:
            seen["gold"] = sandbox
            sandbox.meta.command_count += 1
            return {
                "command": "pytest gold",
                "exit_code": 0,
                "passed": True,
                "output": "exit_code=0",
            }

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", side_effect=fake_run_harness),
            patch.object(runner, "run_gold_test", side_effect=fake_gold),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            record = solve_task("issue-001", harness_version="v0")

        assert seen["harness"] is seen["gold"]
        assert record["sandbox_command_count"] == 3
        assert record["sandbox_cleaned_up"] is True
        assert record["sandbox_backend"] == "docker"
        assert "host" not in str(record["sandbox_backend"]).lower()

    def test_harness_exception_still_saves_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo"),
            patch.object(
                runner,
                "_run_harness",
                side_effect=RuntimeError("planner exploded"),
            ),
            patch.object(
                runner,
                "run_gold_test",
                return_value={
                    "command": "pytest gold",
                    "exit_code": 1,
                    "passed": False,
                    "output": "exit_code=1",
                },
            ),
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            record = solve_task("issue-001", harness_version="v1")

        assert record["success"] is False
        assert record["termination"] == "error"
        assert record["error_type"] == "RuntimeError"
        assert "planner exploded" in record["error_message"]
        path = Path(record["run_path"])
        assert path.name.startswith("issue-001-v1-")
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["error_type"] == "RuntimeError"
        assert record["sandbox_cleaned_up"] is True

    def test_sandbox_start_failure_still_saves_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        FakeSandbox.reset()
        monkeypatch.setattr(runner, "SandboxRunner", FailingStartSandbox)
        task = _fake_task()

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo") as reset_mock,
            patch.object(runner, "_run_harness") as harness_mock,
            patch.object(runner, "run_gold_test") as gold_mock,
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            record = solve_task("issue-001", harness_version="v0")

        reset_mock.assert_called_once()
        harness_mock.assert_not_called()
        gold_mock.assert_not_called()
        assert record["success"] is False
        assert record["termination"] == "error"
        assert record["error_type"] == "SandboxError"
        assert "docker daemon" in record["error_message"]
        assert record["sandbox_backend"] == "docker"
        assert record["sandbox_started"] is False
        assert record["sandbox_cleaned_up"] is False
        assert record.get("sandbox_start_error")
        path = Path(record["run_path"])
        assert path.is_file()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["error_type"] == "SandboxError"

    def test_gold_skipped_when_sandbox_unusable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_sandbox: type[FakeSandbox]
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        task = _fake_task()

        def fake_run_harness(*, sandbox, **kwargs: Any) -> dict[str, Any]:
            sandbox.meta.usable = False
            return _agent_result("v0")

        with (
            patch.object(runner, "get_task", return_value=task),
            patch.object(runner, "resolve_repo_path", return_value=Path(task["repo_path"])),
            patch.object(runner, "reset_repo"),
            patch.object(runner, "_run_harness", side_effect=fake_run_harness),
            patch.object(runner, "run_gold_test") as gold_mock,
            patch.object(runner, "create_client", return_value=object()),
            patch.object(runner, "default_model", return_value="fake-model"),
        ):
            record = solve_task("issue-001", harness_version="v0")

        gold_mock.assert_not_called()
        assert record["success"] is False
        assert record["gold_error_type"] == "SandboxUnusableError"
        assert "skipped gold test" in record["gold_error_message"]
        assert record["sandbox_cleaned_up"] is True


class TestCLI:
    def test_solve_passes_harness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            seen["task_id"] = task_id
            seen.update(kwargs)
            return {
                "task_id": task_id,
                "harness_version": kwargs.get("harness_version"),
                "success": True,
                "difficulty": "easy",
                "termination": "completed",
                "steps": 1,
                "llm_calls": 1,
                "tool_call_count": 0,
                "file_reads": 0,
                "tokens": 1,
                "latency": 0.1,
                "run_path": "runs/x.json",
                "final_answer": "ok",
            }

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        code = cli.main(["solve", "issue-001", "--harness", "v1", "--max-steps", "8"])
        assert code == 0
        assert seen["task_id"] == "issue-001"
        assert seen["harness_version"] == "v1"
        assert seen["max_steps"] == 8
        assert seen["progress"] is not None

    def test_solve_quiet_skips_reporter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {
                "task_id": task_id,
                "harness_version": kwargs.get("harness_version"),
                "success": True,
                "difficulty": "easy",
                "termination": "completed",
                "steps": 1,
                "llm_calls": 1,
                "tool_call_count": 0,
                "file_reads": 0,
                "tokens": 1,
                "latency": 0.1,
                "run_path": "runs/x.json",
                "final_answer": "ok",
            }

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        code = cli.main(["solve", "issue-001", "--quiet"])
        assert code == 0
        assert seen["progress"] is None
        assert seen["interactive_recovery"] is False
        assert seen["require_approval"] is False

    def test_solve_passes_interactive_recovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {
                "task_id": task_id,
                "harness_version": kwargs.get("harness_version"),
                "success": True,
                "difficulty": "easy",
                "termination": "completed",
                "steps": 1,
                "llm_calls": 1,
                "tool_call_count": 0,
                "file_reads": 0,
                "tokens": 1,
                "latency": 0.1,
                "run_path": "runs/x.json",
                "final_answer": "ok",
            }

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        code = cli.main(
            ["solve", "issue-001", "--harness", "v1", "--interactive-recovery"]
        )
        assert code == 0
        assert seen["interactive_recovery"] is True

    def test_compare_runs_both_harnesses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harnesses: list[str] = []
        progress_flags: list[bool] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            harness = kwargs["harness_version"]
            harnesses.append(harness)
            progress_flags.append(kwargs.get("progress") is not None)
            assert "interactive_recovery" not in kwargs
            assert "require_approval" not in kwargs
            return {
                "task_id": task_id,
                "harness_version": harness,
                "success": True,
                "difficulty": "easy",
                "termination": "completed",
                "status": "success" if harness == "v1" else None,
                "workflow_passed": True if harness == "v1" else None,
                "steps": 2,
                "llm_calls": 3,
                "tool_call_count": 1,
                "file_reads": 1,
                "tokens": 10,
                "latency": 0.2,
                "run_path": f"runs/{harness}.json",
                "final_answer": "ok",
            }

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        monkeypatch.setattr(cli, "default_model", lambda: "fake-model")
        code = cli.main(["compare", "issue-001", "--max-steps", "12"])
        assert code == 0
        assert harnesses == ["v0", "v1"]
        assert progress_flags == [True, True]

    def test_compare_quiet_skips_reporter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        flags: list[Any] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            flags.append(kwargs.get("progress"))
            return {
                "task_id": task_id,
                "harness_version": kwargs["harness_version"],
                "success": True,
                "difficulty": "easy",
                "termination": "completed",
                "steps": 1,
                "llm_calls": 1,
                "tool_call_count": 0,
                "file_reads": 0,
                "tokens": 1,
                "latency": 0.1,
                "run_path": "runs/x.json",
                "final_answer": "ok",
            }

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        monkeypatch.setattr(cli, "default_model", lambda: "fake-model")
        code = cli.main(["compare", "issue-001", "--quiet"])
        assert code == 0
        assert flags == [None, None]

    def test_solve_accepts_v2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            seen["task_id"] = task_id
            return {
                "task_id": task_id,
                "harness_version": kwargs.get("harness_version"),
                "success": True,
                "difficulty": "easy",
                "termination": "completed",
                "steps": 1,
                "llm_calls": 1,
                "tool_call_count": 0,
                "file_reads": 0,
                "tokens": 1,
                "latency": 0.1,
                "run_path": "runs/x.json",
                "final_answer": "ok",
            }

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        code = cli.main(
            [
                "solve",
                "issue-009",
                "--harness",
                "v2",
                "--embedder",
                "fastembed",
                "--query-mode",
                "issue+analysis",
            ]
        )
        assert code == 0
        assert seen["harness_version"] == "v2"
        assert seen["embedder_name"] == "fastembed"
        assert seen["query_mode"] == "issue+analysis"

    def test_solve_v2_defaults_hashing_and_issue_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {
                "task_id": task_id,
                "harness_version": kwargs.get("harness_version"),
                "success": True,
                "difficulty": "easy",
                "termination": "completed",
                "steps": 1,
                "llm_calls": 1,
                "tool_call_count": 0,
                "file_reads": 0,
                "tokens": 1,
                "latency": 0.1,
                "run_path": "runs/x.json",
                "final_answer": "ok",
            }

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        code = cli.main(["solve", "issue-009", "--harness", "v2"])
        assert code == 0
        assert seen["embedder_name"] == "hashing"
        assert seen["query_mode"] == "issue"

    def test_rejects_invalid_harness_arg(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["solve", "issue-001", "--harness", "v9"])

    def test_retrieve_requires_task_or_split(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["retrieve"])

    def test_retrieve_wires_eval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_eval(**kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {
                "split": kwargs.get("split"),
                "k": kwargs.get("k"),
                "embedder": "HashingEmbedder",
                "tasks": [
                    {
                        "task_id": "issue-009",
                        "issue": "Ordering 50 widgets",
                        "expected_files": ["app/inventory.py", "app/orders.py"],
                        "modes": {
                            mode: {
                                "recall_at_k": 1.0,
                                "retrieved_files": ["app/inventory.py"],
                            }
                            for mode in ("grep", "bm25", "dense", "hybrid")
                        },
                        "run_path": "runs/issue-009-retrieve-fake.json",
                    }
                ],
                "mean_recall_at_k": {
                    "grep": 1.0,
                    "bm25": 1.0,
                    "dense": 1.0,
                    "hybrid": 1.0,
                },
            }

        monkeypatch.setattr(cli, "run_retrieval_eval", fake_eval)
        code = cli.main(
            ["retrieve", "--split", "hard", "--embedder", "hashing", "--k", "5"]
        )
        assert code == 0
        assert seen["split"] == "hard"
        assert seen["task_id"] is None
        assert seen["embedder_name"] == "hashing"
        assert seen["k"] == 5
        assert seen["reset"] is True
        assert seen["query_mode"] == "issue"

    def test_report_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_report(**kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {
                "solve_cohorts": [],
                "retrieval": [],
                "filters": kwargs,
                "harness_git_sha": "abc",
            }

        monkeypatch.setattr("eval.report.build_report", fake_report)
        code = cli.main(
            [
                "report",
                "--split",
                "hard",
                "--base-commit",
                "deadbeef",
                "--latest-per-cell",
                "--json",
            ]
        )
        assert code == 0
        assert seen["split"] == "hard"
        assert seen["base_commit"] == "deadbeef"
        assert seen["latest_per_cell"] is True

    def test_sandbox_doctor_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Report:
            ok = False
            docker_path = None
            daemon_reachable = False
            server_os = None
            linux_containers = False
            image = "issue-pilot-sandbox:py312"
            image_present = False
            dockerfile_present = True
            requirements_present = True
            warnings: list[str] = []
            errors = ["Docker CLI not found"]

            def to_dict(self) -> dict[str, Any]:
                return {"ok": False, "errors": self.errors}

        monkeypatch.setattr(cli, "doctor", lambda **kwargs: Report())
        code = cli.main(["sandbox", "doctor"])
        assert code == 1


def _cli_solve_record(**overrides: Any) -> dict[str, Any]:
    record = {
        "task_id": "issue-001",
        "harness_version": "v1",
        "success": True,
        "difficulty": "easy",
        "termination": "completed",
        "steps": 1,
        "llm_calls": 1,
        "tool_call_count": 0,
        "file_reads": 0,
        "tokens": 1,
        "latency": 0.1,
        "run_path": "runs/x.json",
        "final_answer": "ok",
    }
    record.update(overrides)
    return record


def _cli_session(**overrides: str) -> Any:
    from eval.session import RunSession

    payload = {
        "run_id": "issue-001-v1-demo",
        "thread_id": "issue-001-v1-demo",
        "task_id": "issue-001",
        "harness": "v1",
        "model": "fake-model",
        "embedder_name": "hashing",
        "query_mode": "issue",
        "repo_path": "C:/fake/benchmark",
        "base_commit": "abc123",
        "status": "paused",
        "created_at": "2026-08-17T00:00:00Z",
    }
    payload.update(overrides)
    return RunSession(**payload)


class TestCLIApproval:
    def test_solve_passes_require_approval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return _cli_solve_record(
                task_id=task_id, harness_version=kwargs.get("harness_version")
            )

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        code = cli.main(
            ["solve", "issue-001", "--harness", "v1", "--require-approval"]
        )
        assert code == 0
        assert seen["require_approval"] is True

    def test_pause_on_approval_implies_require_approval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return _cli_solve_record(
                task_id=task_id, harness_version=kwargs.get("harness_version")
            )

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        code = cli.main(
            ["solve", "issue-001", "--harness", "v1", "--pause-on-approval"]
        )
        assert code == 0
        assert seen["require_approval"] is True

    def test_require_approval_rejects_v0(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["solve", "issue-001", "--require-approval"])

    def test_paused_solve_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "task_id": task_id,
                "harness_version": "v1",
                "paused": True,
                "run_id": "issue-001-v1-demo",
                "status": "waiting_approval",
                "session_path": "runs/sessions/issue-001-v1-demo.json",
                "resume_count": 0,
                "approval_payload": {
                    "issue": "Expired JWT returns 500",
                    "plan": {
                        "problem": "p",
                        "hypothesis": "h",
                        "files_to_inspect": [],
                        "steps": ["a"],
                    },
                    "changed_files": ["app/auth.py"],
                    "git_diff": "diff --git a/app/auth.py",
                    "test_result": {"deterministic_pass": True},
                    "evaluator_result": {"issue_resolved": True},
                },
            }

        monkeypatch.setattr(cli, "solve_task", fake_solve)
        code = cli.main(
            ["solve", "issue-001", "--harness", "v1", "--require-approval"]
        )
        assert code == 0

    def test_runs_lists_paused_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli,
            "list_sessions",
            lambda: [
                _cli_session(),
                _cli_session(run_id="done-run", thread_id="done-run", status="completed"),
            ],
        )
        code = cli.main(["runs"])
        assert code == 0

    def test_review_loads_checkpoint_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, Any] = {}
        monkeypatch.setattr(cli, "load_session", lambda run_id: _cli_session(run_id=run_id, thread_id=run_id))

        def fake_payload(run_id: str, **kwargs: Any) -> dict[str, Any]:
            seen["run_id"] = run_id
            return {
                "issue": "Expired JWT returns 500",
                "plan": {"problem": "p", "hypothesis": "h", "files_to_inspect": [], "steps": ["a"]},
                "changed_files": ["app/auth.py"],
                "git_diff": "diff --git a/app/auth.py",
                "test_result": {"deterministic_pass": True},
                "evaluator_result": {"issue_resolved": True},
            }

        monkeypatch.setattr(cli, "load_review_payload", fake_payload)
        code = cli.main(["review", "issue-001-v1-demo"])
        assert code == 0
        assert seen["run_id"] == "issue-001-v1-demo"

    def test_resume_approve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_resume(run_id: str, **kwargs: Any) -> dict[str, Any]:
            seen["run_id"] = run_id
            seen.update(kwargs)
            return _cli_solve_record(success=True, approval_decision="approve")

        monkeypatch.setattr(cli, "resume_task", fake_resume)
        code = cli.main(["resume", "issue-001-v1-demo", "--approve"])
        assert code == 0
        assert seen["run_id"] == "issue-001-v1-demo"
        assert seen["decision"] == "approve"
        assert seen["feedback"] is None
        assert seen["progress"] is not None

    def test_resume_reject_and_feedback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[dict[str, Any]] = []

        def fake_resume(run_id: str, **kwargs: Any) -> dict[str, Any]:
            seen.append({"run_id": run_id, **kwargs})
            return _cli_solve_record(success=False, approval_decision=kwargs["decision"])

        monkeypatch.setattr(cli, "resume_task", fake_resume)
        assert cli.main(["resume", "issue-001-v1-demo", "--reject"]) == 2
        assert cli.main(
            ["resume", "issue-001-v1-demo", "--feedback", "Drop the unrelated edit"]
        ) == 2
        assert seen[0]["decision"] == "reject"
        assert seen[1]["decision"] == "feedback"
        assert seen[1]["feedback"] == "Drop the unrelated edit"

    def test_resume_requires_decision(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["resume", "issue-001-v1-demo"])

    def test_resume_empty_feedback_errors(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["resume", "issue-001-v1-demo", "--feedback", "  "])


class TestCLIMCP:
    def test_serve_uses_repo_and_does_not_print(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def fake_serve(repo) -> None:
            seen["repo"] = Path(repo)

        monkeypatch.setattr("harness.mcp_server.run_stdio_server", fake_serve)
        code = cli.main(["mcp", "serve", "--repo", str(tmp_path)])
        assert code == 0
        assert seen["repo"] == tmp_path.resolve()

    def test_demo_prints_capability_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def fake_demo(repo, *, path: str, query: str) -> dict[str, Any]:
            assert Path(repo) == tmp_path.resolve()
            assert path == "app/hello.py"
            return {
                "tools": ["read_file", "search_code", "git_diff"],
                "calls": [
                    {
                        "name": "read_file",
                        "arguments": {"path": path},
                        "text": "def greet():\n    return 1\n",
                        "isError": False,
                    }
                ],
            }

        monkeypatch.setattr("harness.mcp_client.run_demo", fake_demo)
        code = cli.main(
            ["mcp", "demo", "--repo", str(tmp_path), "--path", "app/hello.py"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "MCP Client" in out
        assert "read_file" in out
        assert "def greet" in out

    def test_missing_repo_exits_one(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-repo"
        assert cli.main(["mcp", "serve", "--repo", str(missing)]) == 1

