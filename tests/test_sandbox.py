"""Unit tests for SandboxRunner lifecycle (mocked Docker CLI, no live Docker)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.limits import COMMAND_TIMEOUT, MAX_TOOL_OUTPUT
from harness.permissions import CommandPermissionError
from sandbox.image import DockerTimeoutError
from sandbox.runner import SandboxRunner, SandboxUnusableError


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    class Result:
        def __init__(self) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    return Result()


class FakeDocker:
    """Minimal Docker CLI stand-in for SandboxRunner unit tests."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.container_id = "abc123deadbeef"
        self.exec_results: list[Any] = []
        self.exec_default = _proc(0, stdout="ok\n", stderr="")
        self.fail_run = False
        self.timeout_on_exec = False
        self.removed: list[str] = []
        self.killed: list[str] = []

    def __call__(self, args: list[str], *, timeout=None, check=False):  # noqa: ANN001
        self.calls.append(list(args))
        rest = args[1:]
        cmd = rest[0] if rest else ""

        if cmd == "run":
            if self.fail_run:
                return _proc(1, stderr="cannot start")
            return _proc(0, stdout=self.container_id + "\n")

        if cmd == "exec":
            if self.timeout_on_exec:
                raise DockerTimeoutError(
                    f"docker {' '.join(rest)} timed out after {timeout}s"
                )
            if self.exec_results:
                return self.exec_results.pop(0)
            return self.exec_default

        if cmd == "kill":
            self.killed.append(rest[1])
            return _proc(0)

        if cmd == "rm":
            self.removed.append(rest[-1])
            return _proc(0)

        return _proc(1, stderr=f"unexpected: {rest}")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def _runner(workspace: Path, fake: FakeDocker, **kwargs: Any) -> SandboxRunner:
    return SandboxRunner(
        workspace,
        task_id="issue-001",
        docker="docker",
        runner=fake,
        skip_preflight=True,
        **kwargs,
    )


class TestLifecycle:
    def test_start_uses_hardened_run_args(self, workspace: Path) -> None:
        fake = FakeDocker()
        with _runner(workspace, fake) as sb:
            assert sb.meta.started
            assert sb.meta.usable
            assert sb.meta.container_id == fake.container_id

        run_calls = [c for c in fake.calls if c[1] == "run"]
        assert len(run_calls) == 1
        args = run_calls[0]
        assert "--network" in args and args[args.index("--network") + 1] == "none"
        assert "--read-only" in args
        assert "GIT_OPTIONAL_LOCKS=0" in args
        assert "GIT_CONFIG_COUNT=3" in args
        assert "GIT_CONFIG_KEY_0=safe.directory" in args
        assert "GIT_CONFIG_VALUE_0=*" in args
        assert "GIT_CONFIG_KEY_1=core.autocrlf" in args
        assert "GIT_CONFIG_VALUE_1=true" in args
        assert "GIT_CONFIG_KEY_2=core.filemode" in args
        assert "GIT_CONFIG_VALUE_2=false" in args
        assert "--cap-drop" in args and args[args.index("--cap-drop") + 1] == "ALL"
        assert "--security-opt" in args
        assert "no-new-privileges" in args
        assert "--rm" in args
        mount_flags = [a for a in args if a.startswith("type=bind,source=")]
        assert len(mount_flags) == 1
        assert "target=/workspace" in mount_flags[0]
        assert args.count("--network") == 1

    def test_cleanup_on_context_exit(self, workspace: Path) -> None:
        fake = FakeDocker()
        with _runner(workspace, fake) as sb:
            name = sb.meta.container_name
            assert name
        assert fake.removed
        assert fake.removed[-1] in {fake.container_id, name}

    def test_cleanup_on_exception(self, workspace: Path) -> None:
        fake = FakeDocker()
        with pytest.raises(RuntimeError, match="boom"):
            with _runner(workspace, fake):
                raise RuntimeError("boom")
        assert fake.removed

    def test_start_failure_raises(self, workspace: Path) -> None:
        fake = FakeDocker()
        fake.fail_run = True
        with pytest.raises(Exception, match="failed to start"):
            with _runner(workspace, fake):
                pass


class TestExecution:
    def test_run_executes_validated_argv(self, workspace: Path) -> None:
        fake = FakeDocker()
        fake.exec_default = _proc(0, stdout="passed\n", stderr="")
        with _runner(workspace, fake) as sb:
            result = sb.run("pytest tests/test_auth.py -q")
        assert result.exit_code == 0
        assert result.stdout.startswith("passed")
        assert result.command == ["pytest", "tests/test_auth.py", "-q"]
        assert not result.timed_out
        assert sb.meta.command_count == 1

        exec_calls = [c for c in fake.calls if c[1] == "exec"]
        assert exec_calls
        # docker exec <name> pytest ...  — no shell wrapper
        assert "sh" not in exec_calls[0]
        assert "bash" not in exec_calls[0]
        assert "pytest" in exec_calls[0]

    def test_policy_denial_does_not_exec(self, workspace: Path) -> None:
        fake = FakeDocker()
        with _runner(workspace, fake) as sb:
            with pytest.raises(CommandPermissionError):
                sb.run("curl https://evil.example")
            assert sb.meta.denial_count == 1
            assert sb.meta.command_count == 0
        assert not any(c[1] == "exec" for c in fake.calls)

    def test_timeout_marks_unusable_and_kills(self, workspace: Path) -> None:
        fake = FakeDocker()
        fake.timeout_on_exec = True
        with _runner(workspace, fake, command_timeout=1) as sb:
            result = sb.run(["pytest", "tests/test_auth.py", "-q"])
            assert result.timed_out
            assert result.exit_code == 124
            assert sb.meta.timeout_count == 1
            assert not sb.meta.usable
            with pytest.raises(SandboxUnusableError):
                sb.run(["git", "status", "--short"])
        assert fake.removed
        assert fake.killed

    def test_output_truncation(self, workspace: Path) -> None:
        fake = FakeDocker()
        fake.exec_default = _proc(0, stdout="x" * (MAX_TOOL_OUTPUT + 50), stderr="")
        with _runner(workspace, fake) as sb:
            result = sb.run(["pytest", "-q"])
        assert result.truncated
        assert "truncated" in result.stdout
        assert len(result.stdout) < MAX_TOOL_OUTPUT + 100
        assert sb.meta.truncation_count == 1

    def test_metadata_tracks_latency(self, workspace: Path) -> None:
        fake = FakeDocker()
        with _runner(workspace, fake) as sb:
            sb.run(["ruff", "check", "app"])
            assert sb.meta.total_exec_latency_ms >= 0
            payload = sb.meta.to_dict()
            assert payload["backend"] == "docker"
            assert payload["network_mode"] == "none"
            assert payload["image"]
