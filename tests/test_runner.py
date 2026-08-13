"""Mocked tests for versioned runner dispatch and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import cli
import eval.runner as runner
from eval.runner import _normalize_harness, save_run, solve_task
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
    if harness == "v1":
        base.update(
            {
                "analysis": "analysis",
                "plan": {"problem": "p", "hypothesis": "h", "files_to_inspect": [], "steps": ["a"]},
                "diagnosis": "",
                "test_result": {"exit_code": 0, "passed": True},
                "status": "success",
                "workflow_passed": True,
            }
        )
    return base


class TestHarnessNormalization:
    def test_accepts_v0_v1(self) -> None:
        assert _normalize_harness("v0") == "v0"
        assert _normalize_harness("V1") == "v1"

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="harness_version"):
            _normalize_harness("v2")


class TestSaveRun:
    def test_filename_includes_harness(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
        path = save_run({"task_id": "issue-001", "harness_version": "v1", "ok": True})
        assert path.name.startswith("issue-001-v1-")
        assert path.suffix == ".json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["harness_version"] == "v1"


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
        ):
            record = solve_task("issue-001", harness_version="v0", max_steps=9)

        assert calls == ["v0"]
        reset_mock.assert_called_once()
        assert record["harness_version"] == "v0"
        assert record["success"] is True
        assert record["llm_calls"] == 4
        assert record["tokens"] == 120
        assert "analysis" not in record
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
        assert record["harness_version"] == "v1"
        assert record["analysis"] == "analysis"
        assert record["plan"]["steps"] == ["a"]
        assert record["workflow_passed"] is True
        assert record["verification"]["exit_code"] == 0
        assert Path(record["run_path"]).name.startswith("issue-001-v1-")

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

    def test_compare_runs_both_harnesses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harnesses: list[str] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            harness = kwargs["harness_version"]
            harnesses.append(harness)
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

    def test_rejects_invalid_harness_arg(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["solve", "issue-001", "--harness", "v9"])

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
