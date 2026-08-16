"""Live Docker integration tests for SandboxRunner.

Skipped automatically when Docker Desktop or the sandbox image is unavailable.
Run explicitly with: pytest tests/test_sandbox_docker.py -m docker
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from eval.repository import find_host_git
from harness.limits import MAX_TOOL_OUTPUT
from harness.permissions import CommandPermissionError
from sandbox.image import run_docker
from sandbox.runner import SandboxRunner, SandboxUnusableError

pytestmark = pytest.mark.docker


def _inspect(name: str, fmt: str) -> str:
    proc = run_docker(["inspect", "-f", fmt, name], timeout=30)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return (proc.stdout or "").strip()


def _container_exists(name: str) -> bool:
    proc = run_docker(["inspect", name], timeout=30)
    return proc.returncode == 0


def _seed_git(workspace: Path) -> None:
    git = find_host_git()
    if git is None:
        pytest.skip("host git is required to seed the workspace")
    for args in (
        ["init"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
        ["add", "."],
        ["commit", "-m", "init"],
    ):
        subprocess.run(
            [git, "-C", str(workspace), *args],
            check=True,
            capture_output=True,
            text=True,
        )


def _verify_config(workspace: Path, sandbox: SandboxRunner) -> dict:
    return {
        "configurable": {
            "repo_path": str(workspace),
            "test_command": "pytest tests/test_ok.py -q",
            "lint_command": "ruff check app --ignore EXE001 --ignore EXE002",
            "sandbox": sandbox,
        }
    }


@pytest.fixture
def live_workspace(tmp_path: Path) -> Path:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_ok.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def test_ok():\n"
        "    assert Path('/workspace/visible.txt').read_text() == 'mounted-ok\\n'\n",
        encoding="utf-8",
    )
    (tests / "test_sleep.py").write_text(
        "import time\n\ndef test_sleep():\n    time.sleep(120)\n",
        encoding="utf-8",
    )
    (tests / "test_loud.py").write_text(
        "def test_loud():\n    print('x' * 25000)\n",
        encoding="utf-8",
    )
    (tmp_path / "visible.txt").write_text("mounted-ok\n", encoding="utf-8")
    app = tmp_path / "app"
    app.mkdir()
    (app / "hello.py").write_text(
        "#!/usr/bin/env python3\n"
        'def greet(name: str) -> str:\n    return f"hi {name}"\n',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def live_sandbox(docker_preflight, live_workspace: Path):
    with SandboxRunner(live_workspace, task_id="itest") as sb:
        yield sb


class TestDockerSandbox:
    def test_workspace_mount_and_pytest(self, live_sandbox: SandboxRunner) -> None:
        result = live_sandbox.run(["pytest", "tests/test_ok.py", "-q"])
        assert result.exit_code == 0
        assert not result.timed_out
        combined = result.stdout + result.stderr
        assert "passed" in combined or result.exit_code == 0

    def test_network_disabled_and_only_workspace_bind(
        self, live_sandbox: SandboxRunner
    ) -> None:
        name = live_sandbox.meta.container_name
        assert name
        network = _inspect(name, "{{.HostConfig.NetworkMode}}")
        assert network == "none"

        mounts = json.loads(_inspect(name, "{{json .Mounts}}") or "[]")
        binds = [m for m in mounts if m.get("Type") == "bind"]
        assert len(binds) == 1
        assert binds[0].get("Destination") == "/workspace"
        destinations = {m.get("Destination") for m in binds}
        assert "/etc" not in destinations
        assert "/var/run/docker.sock" not in destinations

        with pytest.raises(CommandPermissionError):
            live_sandbox.run("curl https://example.com")

    def test_output_truncation(self, live_sandbox: SandboxRunner) -> None:
        result = live_sandbox.run(["pytest", "tests/test_loud.py", "-s", "-q"])
        assert result.truncated
        assert "truncated" in result.stdout or "truncated" in result.stderr
        assert len(result.stdout) <= MAX_TOOL_OUTPUT + 80

    def test_timeout_then_unusable(self, docker_preflight, live_workspace: Path) -> None:
        with SandboxRunner(
            live_workspace, task_id="itest-timeout", command_timeout=3
        ) as sb:
            name = sb.meta.container_name
            result = sb.run(["pytest", "tests/test_sleep.py", "-q"])
            assert result.timed_out
            assert result.exit_code == 124
            with pytest.raises(SandboxUnusableError):
                sb.run(["pytest", "tests/test_ok.py", "-q"])
        assert name
        assert not _container_exists(name)

    def test_cleanup_after_normal_exit(
        self, docker_preflight, live_workspace: Path
    ) -> None:
        with SandboxRunner(live_workspace, task_id="itest-normal") as sb:
            name = sb.meta.container_name
            sb.run(["pytest", "tests/test_ok.py", "-q"])
            assert name and _container_exists(name)
        assert name
        assert not _container_exists(name)

    def test_cleanup_after_exception(
        self, docker_preflight, live_workspace: Path
    ) -> None:
        name = None
        with pytest.raises(RuntimeError, match="boom"):
            with SandboxRunner(live_workspace, task_id="itest-exc") as sb:
                name = sb.meta.container_name
                raise RuntimeError("boom")
        assert name
        assert not _container_exists(name)

    def test_git_diff_head_inside_container(
        self, docker_preflight, live_workspace: Path
    ) -> None:
        _seed_git(live_workspace)
        (live_workspace / "visible.txt").write_text("changed\n", encoding="utf-8")

        from agent.tools.git import git_diff

        with SandboxRunner(live_workspace, task_id="itest-git") as sb:
            diff = sb.run(["git", "diff", "HEAD"])
            status = sb.run(["git", "status", "--short"])
            out = git_diff(live_workspace, sandbox=sb)
        assert diff.exit_code in (0, 1), diff.stderr or diff.stdout
        assert status.exit_code == 0, status.stderr or status.stdout
        assert not out.startswith("Error:")
        assert "visible.txt" in out

    def test_layer1_pass_runs_pytest_ruff_and_git(
        self, docker_preflight, live_workspace: Path
    ) -> None:
        _seed_git(live_workspace)
        (live_workspace / "app" / "hello.py").write_text(
            "#!/usr/bin/env python3\n"
            'def greet(name: str) -> str:\n    return f"hey {name}"\n',
            encoding="utf-8",
        )
        from agent.nodes.verify import deterministic_verify
        from agent.state import initial_state

        with SandboxRunner(live_workspace, task_id="itest-l1-pass") as sb:
            before = sb.meta.command_count
            orig = sb.run
            argv_log: list[list[str]] = []

            def tracking(command):
                result = orig(command)
                argv_log.append(list(result.command))
                return result

            sb.run = tracking  # type: ignore[method-assign]
            out = deterministic_verify(
                initial_state("bug"),
                _verify_config(live_workspace, sb),
            )
            after = sb.meta.command_count

        result = out["test_result"]
        assert after - before >= 4
        assert any(argv and argv[0] == "pytest" for argv in argv_log)
        assert any(argv and argv[0] == "ruff" for argv in argv_log)
        assert any(argv[:2] == ["git", "diff"] for argv in argv_log)
        assert any(argv[:2] == ["git", "status"] for argv in argv_log)
        assert result["pytest_passed"] is True, result["output"]
        assert result["ruff_passed"] is True, result["ruff_output"]
        assert result["patch_valid"] is True
        assert result["deterministic_pass"] is True

    def test_layer1_clean_tree_fails_patch_gate(
        self, docker_preflight, live_workspace: Path
    ) -> None:
        _seed_git(live_workspace)
        from agent.nodes.verify import deterministic_verify
        from agent.state import initial_state

        with SandboxRunner(live_workspace, task_id="itest-l1-clean") as sb:
            status = sb.run(["git", "status", "--short"])
            if (status.stdout or "").strip():
                pytest.skip(
                    "sandbox mount dirtied the worktree after commit "
                    f"({status.stdout!r})"
                )
            out = deterministic_verify(
                initial_state("bug"),
                _verify_config(live_workspace, sb),
            )

        result = out["test_result"]
        assert result["pytest_passed"] is True, result["output"]
        assert result["ruff_passed"] is True, result["ruff_output"]
        assert result["patch_valid"] is False
        assert result["deterministic_pass"] is False
        assert result["git_diff"] == "(no changes)"

    def test_crlf_only_worktree_is_not_a_patch(
        self, docker_preflight, tmp_path: Path
    ) -> None:
        git = find_host_git()
        if git is None:
            pytest.skip("host git is required to seed the workspace")
        tests = tmp_path / "tests"
        tests.mkdir()
        app = tmp_path / "app"
        app.mkdir()
        (tmp_path / "visible.txt").write_bytes(b"mounted-ok\n")
        (tests / "test_ok.py").write_bytes(
            b"from pathlib import Path\n"
            b"\n"
            b"def test_ok():\n"
            b"    assert Path('/workspace/visible.txt').read_text() == 'mounted-ok\\n'\n"
        )
        hello = (
            b"#!/usr/bin/env python3\n"
            b"def greet(name: str) -> str:\n"
            b'    return f"hi {name}"\n'
        )
        (app / "hello.py").write_bytes(hello.replace(b"\n", b"\r\n"))
        for args in (
            ["init"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "Test"],
            ["config", "core.autocrlf", "true"],
            ["config", "core.filemode", "false"],
            ["add", "."],
            ["commit", "-m", "init"],
        ):
            subprocess.run(
                [git, "-C", str(tmp_path), *args],
                check=True,
                capture_output=True,
                text=True,
            )

        from agent.nodes.verify import deterministic_verify
        from agent.state import initial_state

        with SandboxRunner(tmp_path, task_id="itest-crlf") as sb:
            status = sb.run(["git", "status", "--short"])
            assert (status.stdout or "").strip() == "", status.stdout
            out = deterministic_verify(
                initial_state("bug"),
                _verify_config(tmp_path, sb),
            )

        result = out["test_result"]
        assert result["pytest_passed"] is True, result["output"]
        assert result["ruff_passed"] is True, result["ruff_output"]
        assert result["changed_files"] == []
        assert result["untracked_files"] == []
        assert result["patch_valid"] is False
        assert result["deterministic_pass"] is False
        assert result["git_diff"] == "(no changes)"
