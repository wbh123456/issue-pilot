"""Unit tests for agent tools — uses an isolated temp repo, not the benchmark."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agent.tools import (
    edit_file,
    git_diff,
    grep_code,
    list_files,
    read_file,
    run_tests,
)
from agent.tools._sandbox import MAX_TOOL_OUTPUT, truncate_output
from agent.tools.git import _find_git


def _git(repo: Path, *args: str) -> None:
    git = _find_git()
    assert git is not None, "git is required for these tests"
    subprocess.run(
        [git, "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


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


class TestGrep:
    def test_grep_finds_match(self, repo: Path) -> None:
        out = grep_code(repo, "greet")
        assert "hello.py" in out
        assert "greet" in out

    def test_grep_no_match(self, repo: Path) -> None:
        assert grep_code(repo, "zzz_no_such_token") == "(no matches)"

    def test_grep_empty_query(self, repo: Path) -> None:
        assert grep_code(repo, "").startswith("Error:")


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
    def test_run_python_command(self, repo: Path) -> None:
        cmd = f'{sys.executable} -c "print(42)"'
        out = run_tests(repo, cmd)
        assert "exit_code=0" in out
        assert "42" in out

    def test_run_failing_command(self, repo: Path) -> None:
        cmd = f'{sys.executable} -c "raise SystemExit(7)"'
        out = run_tests(repo, cmd)
        assert "exit_code=7" in out

    def test_empty_command(self, repo: Path) -> None:
        assert run_tests(repo, "   ").startswith("Error:")

    def test_pytest_fallback_rewrites_when_missing(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agent.tools.shell.shutil.which", lambda _name: None)
        # `-h` exits 0 quickly and avoids needing a real test suite.
        out = run_tests(repo, "pytest -h")
        assert "exit_code=0" in out
        assert "-m pytest" in out or "pytest" in out


class TestGitDiff:
    def test_clean_tree(self, repo: Path) -> None:
        assert git_diff(repo) == "(no changes)"

    def test_shows_modification(self, repo: Path) -> None:
        edit_file(
            repo,
            "app/hello.py",
            old_str="return f'hi {name}'",
            new_str="return f'hey {name}'",
        )
        out = git_diff(repo)
        assert "hello.py" in out
        assert "hey {name}" in out or "+return" in out

    def test_shows_untracked(self, repo: Path) -> None:
        edit_file(repo, "scratch.txt", content="tmp\n")
        out = git_diff(repo)
        assert "scratch.txt" in out


class TestTruncate:
    def test_truncate_output(self) -> None:
        text = "a" * (MAX_TOOL_OUTPUT + 50)
        out = truncate_output(text)
        assert len(out) < len(text)
        assert "truncated" in out
