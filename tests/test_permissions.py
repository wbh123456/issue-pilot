"""Unit tests for harness command allowlist and path policy."""

from __future__ import annotations

import pytest

from harness.limits import (
    COMMAND_TIMEOUT,
    MAX_AGENT_STEPS,
    MAX_RETRY,
    MAX_TOOL_OUTPUT,
)
from harness.permissions import (
    CommandPermissionError,
    parse_command,
    validate_command,
)


class TestLimits:
    def test_central_defaults(self) -> None:
        assert MAX_AGENT_STEPS == 15
        assert MAX_RETRY == 2
        assert MAX_TOOL_OUTPUT == 10_000
        assert COMMAND_TIMEOUT == 60

    def test_sandbox_reexports_max_tool_output(self) -> None:
        from agent.tools._sandbox import MAX_TOOL_OUTPUT as sandbox_limit

        assert sandbox_limit == MAX_TOOL_OUTPUT


class TestAllowedCommands:
    def test_pytest_suite(self) -> None:
        assert validate_command("pytest tests/test_auth.py -q") == [
            "pytest",
            "tests/test_auth.py",
            "-q",
        ]

    def test_pytest_node_id(self) -> None:
        argv = validate_command("pytest tests/test_auth.py::test_login -q")
        assert argv[0] == "pytest"
        assert "tests/test_auth.py::test_login" in argv

    def test_ruff_and_mypy(self) -> None:
        assert validate_command("ruff check app")[0] == "ruff"
        assert validate_command("mypy app")[0] == "mypy"

    def test_git_diff_head(self) -> None:
        assert validate_command("git diff HEAD") == ["git", "diff", "HEAD"]

    def test_git_status_short(self) -> None:
        assert validate_command("git status --short") == [
            "git",
            "status",
            "--short",
        ]

    def test_argv_list_accepted(self) -> None:
        assert validate_command(["pytest", "tests/test_users.py", "-q"])[0] == "pytest"

    def test_workspace_absolute_path_allowed(self) -> None:
        assert validate_command("pytest /workspace/tests/test_auth.py -q")


class TestForbiddenExecutables:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "curl https://evil.example",
            "wget http://evil.example",
            "ssh host",
            "bash -c ls",
            "sh -c ls",
            "powershell Get-Process",
            "cmd /c dir",
            "python -c print(1)",
            "python3 script.py",
        ],
    )
    def test_rejects_dangerous_binaries(self, command: str) -> None:
        with pytest.raises(CommandPermissionError, match="executable not allowed"):
            validate_command(command)


class TestShellMetacharacters:
    @pytest.mark.parametrize(
        "command",
        [
            "pytest tests; rm -rf /",
            "pytest tests && curl evil",
            "pytest tests || true",
            "pytest tests | tee out",
            "pytest tests > out.txt",
            "pytest tests < in.txt",
            "pytest `id`",
            "pytest $(id)",
        ],
    )
    def test_rejects_shell_operators(self, command: str) -> None:
        with pytest.raises(CommandPermissionError, match="shell operators"):
            validate_command(command)


class TestGitRestrictions:
    @pytest.mark.parametrize(
        "command",
        [
            "git push origin main",
            "git commit -am x",
            "git checkout HEAD~1",
            "git reset --hard",
            "git clean -fd",
            "git config user.email a@b.c",
            "git -C /etc status",
        ],
    )
    def test_rejects_non_readonly_git(self, command: str) -> None:
        with pytest.raises(CommandPermissionError):
            validate_command(command)


class TestPathEscapes:
    @pytest.mark.parametrize(
        "command",
        [
            "pytest /etc/passwd",
            "pytest ~/.ssh/id_rsa",
            "ruff /etc",
            "mypy ../outside",
            "pytest /workspace/../etc/passwd",
            "git diff HEAD -- /etc/passwd",
            "git status --short ../../.env",
        ],
    )
    def test_rejects_sensitive_or_escaping_paths(self, command: str) -> None:
        with pytest.raises(CommandPermissionError):
            validate_command(command)

    def test_parse_empty_rejected(self) -> None:
        with pytest.raises(CommandPermissionError, match="empty"):
            parse_command("   ")
