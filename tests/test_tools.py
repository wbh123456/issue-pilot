"""Unit tests for agent tools — uses an isolated temp repo, not the benchmark."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agent.tools import (
    TOOLS,
    V2_TOOLS,
    edit_file,
    execute_tool,
    git_diff,
    grep_code,
    list_files,
    read_file,
    run_tests,
    search_code,
)
from agent.tools.schema import SEARCH_CODE_TOOL
from agent.tools._sandbox import MAX_TOOL_OUTPUT, truncate_output
from eval.repository import find_host_git
from harness.permissions import validate_command
from sandbox.runner import CommandResult


def _git(repo: Path, *args: str) -> None:
    git = find_host_git()
    assert git is not None, "git is required for these tests"
    subprocess.run(
        [git, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@dataclass
class ScriptedSandbox:
    """Fake SandboxRunner: policy-check, then return scripted CommandResults."""

    results: list[CommandResult] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def run(self, command: str | list[str]) -> CommandResult:
        argv = validate_command(command)
        self.calls.append(argv)
        if not self.results:
            raise AssertionError(f"unexpected sandbox command: {argv}")
        return self.results.pop(0)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Minimal git repo that mirrors the sandbox shape tools expect."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "hello.py").write_text(
        "def greet(name):\n    return f'hi {name}'\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "dup.py").write_text(
        "x = 1\nx = 1\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# toy\n", encoding="utf-8")

    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    # Avoid "master" vs "main" surprises on older git.
    _git(tmp_path, "branch", "-M", "main")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "init")
    return tmp_path


class TestListAndRead:
    def test_list_files_root(self, repo: Path) -> None:
        out = list_files(repo, ".")
        assert "app/" in out
        assert "README.md" in out

    def test_list_files_subdir(self, repo: Path) -> None:
        out = list_files(repo, "app")
        assert "app/hello.py" in out

    def test_list_files_missing(self, repo: Path) -> None:
        assert list_files(repo, "nope").startswith("Error:")

    def test_read_file(self, repo: Path) -> None:
        out = read_file(repo, "app/hello.py")
        assert "def greet" in out

    def test_read_file_missing(self, repo: Path) -> None:
        assert "not found" in read_file(repo, "missing.py")

    def test_read_file_directory(self, repo: Path) -> None:
        assert "directory" in read_file(repo, "app")

    def test_path_escape_rejected(self, repo: Path) -> None:
        with pytest.raises(PermissionError):
            list_files(repo, "../")

    def test_nested_path_escape_rejected(self, repo: Path) -> None:
        with pytest.raises(PermissionError):
            read_file(repo, "app/../../outside.txt")

    def test_absolute_path_outside_rejected(self, repo: Path) -> None:
        outside = Path.cwd().anchor  # drive root on Windows, "/" on POSIX
        with pytest.raises(PermissionError):
            read_file(repo, outside)

    def test_symlink_escape_rejected(self, repo: Path, tmp_path: Path) -> None:
        outside = tmp_path.parent / f"issue_pilot_secret_{tmp_path.name}.txt"
        outside.write_text("secret-leaked\n", encoding="utf-8")
        link = repo / "leak.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlink creation requires privileges on this host")
        try:
            with pytest.raises(PermissionError):
                read_file(repo, "leak.txt")
        finally:
            outside.unlink(missing_ok=True)

    def test_gold_staging_dir_is_invisible(self, repo: Path) -> None:
        gold = repo / "tests" / "_gold"
        gold.mkdir(parents=True)
        (gold / "test_issue_001.py").write_text(
            "def test_expired_token_returns_401():\n    assert True\n",
            encoding="utf-8",
        )
        listed = list_files(repo, "tests")
        assert "_gold" not in listed
        with pytest.raises(PermissionError, match="not visible"):
            read_file(repo, "tests/_gold/test_issue_001.py")
        grep_out = grep_code(repo, "test_expired_token_returns_401")
        assert "_gold" not in grep_out
        assert "test_expired_token_returns_401" not in grep_out


class TestGrep:
    def test_grep_finds_match(self, repo: Path) -> None:
        out = grep_code(repo, "greet")
        assert "hello.py" in out
        assert "greet" in out

    def test_grep_no_match(self, repo: Path) -> None:
        assert grep_code(repo, "zzz_no_such_token") == "(no matches)"

    def test_grep_empty_query(self, repo: Path) -> None:
        assert grep_code(repo, "").startswith("Error:")

    def test_grep_does_not_use_host_ripgrep(self, repo: Path) -> None:
        import agent.tools.search as search_mod

        assert not hasattr(search_mod, "_grep_with_rg")
        out = grep_code(repo, "greet")
        assert "hello.py" in out
        assert "greet" in out


def _tool_names(tools: list) -> list[str]:
    return [t["function"]["name"] for t in tools]


class TestSearchCode:
    def test_v0_v1_tools_exclude_search_code(self) -> None:
        names = _tool_names(TOOLS)
        assert names == [
            "list_files",
            "read_file",
            "grep_code",
            "edit_file",
            "run_tests",
            "git_diff",
        ]
        assert "search_code" not in names
        assert V2_TOOLS is not TOOLS
        assert _tool_names(V2_TOOLS) == [*names, "search_code"]

    def test_description_is_a_starting_locator(self) -> None:
        desc = SEARCH_CODE_TOOL["function"]["description"].lower()
        assert "start here" in desc
        assert "prefer this over grep_code" in desc
        assert "literal grep misses" not in desc

    def test_disabled_by_default(self, repo: Path) -> None:
        out = execute_tool(
            "search_code",
            {"query": "greet"},
            repo_path=str(repo),
            test_command="pytest -q",
        )
        assert out == "Error: unknown tool: search_code"

    def test_enabled_hybrid_search(self, repo: Path) -> None:
        out = execute_tool(
            "search_code",
            {"query": "greet"},
            repo_path=str(repo),
            test_command="pytest -q",
            search_code_enabled=True,
        )
        assert "hello.py" in out
        assert "greet" in out

    def test_empty_query(self, repo: Path) -> None:
        assert search_code(repo, "").startswith("Error:")
        out = execute_tool(
            "search_code",
            {"query": ""},
            repo_path=str(repo),
            test_command="pytest -q",
            search_code_enabled=True,
        )
        assert out.startswith("Error:")

    def test_skips_gold_staging(self, repo: Path) -> None:
        gold = repo / "tests" / "_gold"
        gold.mkdir(parents=True)
        (gold / "secret.py").write_text(
            "def leaked_gold_symbol():\n    return 1\n",
            encoding="utf-8",
        )
        out = search_code(repo, "leaked_gold_symbol")
        assert "_gold" not in out
        assert "leaked_gold_symbol" not in out


class TestEditFile:
    def test_search_replace(self, repo: Path) -> None:
        msg = edit_file(
            repo,
            "app/hello.py",
            old_str="return f'hi {name}'",
            new_str="return f'hello {name}'",
        )
        assert msg.startswith("Updated")
        assert "hello {name}" in read_file(repo, "app/hello.py")

    def test_full_write_creates_file(self, repo: Path) -> None:
        msg = edit_file(repo, "app/new.py", content="VALUE = 1\n")
        assert msg.startswith("Wrote")
        assert read_file(repo, "app/new.py") == "VALUE = 1\n"

    def test_rejects_both_modes(self, repo: Path) -> None:
        out = edit_file(
            repo,
            "app/hello.py",
            content="x",
            old_str="a",
            new_str="b",
        )
        assert out.startswith("Error:")

    def test_rejects_neither_mode(self, repo: Path) -> None:
        assert edit_file(repo, "app/hello.py").startswith("Error:")

    def test_replace_requires_exact_one_match(self, repo: Path) -> None:
        out = edit_file(repo, "app/dup.py", old_str="x = 1", new_str="x = 2")
        assert "2 times" in out


class TestRunTests:
    def test_run_pytest_via_sandbox(self, repo: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(
                    command=["pytest", "tests/test_auth.py", "-q"],
                    exit_code=0,
                    stdout="1 passed",
                    stderr="",
                )
            ]
        )
        out = run_tests(repo, "pytest tests/test_auth.py -q", sandbox=sandbox)
        assert "exit_code=0" in out
        assert "1 passed" in out
        assert sandbox.calls == [["pytest", "tests/test_auth.py", "-q"]]

    def test_run_failing_command(self, repo: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(
                    command=["pytest", "-q"],
                    exit_code=7,
                    stdout="",
                    stderr="failed",
                )
            ]
        )
        out = run_tests(repo, "pytest -q", sandbox=sandbox)
        assert "exit_code=7" in out
        assert "failed" in out

    def test_empty_command(self, repo: Path) -> None:
        sandbox = ScriptedSandbox()
        assert run_tests(repo, "   ", sandbox=sandbox).startswith("Error:")
        assert sandbox.calls == []

    def test_requires_sandbox(self, repo: Path) -> None:
        with pytest.raises(RuntimeError, match="host execution is not allowed"):
            run_tests(repo, "pytest -q", sandbox=None)  # type: ignore[arg-type]

    def test_rejects_host_python(self, repo: Path) -> None:
        sandbox = ScriptedSandbox()
        out = run_tests(repo, 'python -c "print(42)"', sandbox=sandbox)
        assert out.startswith("Error: command not allowed")
        assert sandbox.calls == []

    def test_timeout_message(self, repo: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(
                    command=["pytest", "-q"],
                    exit_code=124,
                    stdout="",
                    stderr="timed out",
                    timed_out=True,
                )
            ]
        )
        out = run_tests(repo, "pytest -q", sandbox=sandbox)
        assert out.startswith("Error: tests timed out")


class TestGitDiff:
    def test_clean_tree(self, repo: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(["git", "diff", "HEAD"], 0, "", ""),
                CommandResult(["git", "status", "--short"], 0, "", ""),
            ]
        )
        assert git_diff(repo, sandbox=sandbox) == "(no changes)"
        assert sandbox.calls == [
            ["git", "diff", "HEAD"],
            ["git", "status", "--short"],
        ]

    def test_shows_modification(self, repo: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(
                    ["git", "diff", "HEAD"],
                    0,
                    "diff --git a/app/hello.py b/app/hello.py\n+return f'hey {name}'\n",
                    "",
                ),
                CommandResult(["git", "status", "--short"], 0, " M app/hello.py\n", ""),
            ]
        )
        out = git_diff(repo, sandbox=sandbox)
        assert "hello.py" in out
        assert "hey {name}" in out or "+return" in out

    def test_shows_untracked(self, repo: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(["git", "diff", "HEAD"], 0, "", ""),
                CommandResult(["git", "status", "--short"], 0, "?? scratch.txt\n", ""),
            ]
        )
        out = git_diff(repo, sandbox=sandbox)
        assert "scratch.txt" in out

    def test_requires_sandbox(self, repo: Path) -> None:
        with pytest.raises(RuntimeError, match="host execution is not allowed"):
            git_diff(repo, sandbox=None)  # type: ignore[arg-type]

    def test_nonzero_exit_is_one_line_without_usage_dump(self, repo: Path) -> None:
        usage = (
            "warning: Not a git repository. Use --no-index to compare two paths\n"
            "usage: git diff --no-index [<options>] <path> <path>\n"
            + ("x" * 4000)
        )
        sandbox = ScriptedSandbox(
            [
                CommandResult(["git", "diff", "HEAD"], 129, "", usage),
                CommandResult(["git", "status", "--short"], 0, "", ""),
            ]
        )
        out = git_diff(repo, sandbox=sandbox)
        assert out.startswith("Error: git diff failed (exit 129):")
        assert "Not a git repository" in out
        assert "\n" not in out
        assert "usage:" not in out
        assert "xxxx" not in out
        assert len(out) < 300


class TestExecuteToolSandbox:
    def test_run_tests_and_git_diff_inject_sandbox(self, repo: Path) -> None:
        sandbox = ScriptedSandbox(
            [
                CommandResult(["pytest", "-q"], 0, "ok", ""),
                CommandResult(["git", "diff", "HEAD"], 0, "", ""),
                CommandResult(["git", "status", "--short"], 0, "", ""),
            ]
        )
        tests_out = execute_tool(
            "run_tests",
            {},
            repo_path=str(repo),
            test_command="pytest -q",
            sandbox=sandbox,
        )
        diff_out = execute_tool(
            "git_diff",
            {},
            repo_path=str(repo),
            test_command="pytest -q",
            sandbox=sandbox,
        )
        assert "exit_code=0" in tests_out
        assert diff_out == "(no changes)"

    def test_no_host_fallback_without_sandbox(self, repo: Path) -> None:
        with pytest.raises(RuntimeError, match="host execution is not allowed"):
            execute_tool(
                "run_tests",
                {},
                repo_path=str(repo),
                test_command="pytest -q",
                sandbox=None,
            )


class TestTruncate:
    def test_truncate_output(self) -> None:
        text = "a" * (MAX_TOOL_OUTPUT + 50)
        out = truncate_output(text)
        assert len(out) < len(text)
        assert "truncated" in out
