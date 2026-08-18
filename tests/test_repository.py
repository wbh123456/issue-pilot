"""Host git helpers used by evaluator reset and resume worktree checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eval.repository import (
    WorktreeError,
    find_host_git,
    git_sha,
    verify_resume_worktree,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    git = find_host_git()
    if git is None:
        pytest.skip("git executable not found")
    return subprocess.run(
        [git, "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    git = find_host_git()
    if git is None:
        pytest.skip("git executable not found")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "base")
    sha = git_sha(repo)
    assert sha
    return repo, sha


class TestVerifyResumeWorktree:
    def test_accepts_dirty_tree_on_base_commit(self, tmp_path: Path) -> None:
        repo, sha = _init_repo(tmp_path)
        (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
        verify_resume_worktree(repo, sha)

    def test_rejects_clean_worktree(self, tmp_path: Path) -> None:
        repo, sha = _init_repo(tmp_path)
        with pytest.raises(WorktreeError, match="worktree is clean"):
            verify_resume_worktree(repo, sha)

    def test_rejects_moved_head(self, tmp_path: Path) -> None:
        repo, sha = _init_repo(tmp_path)
        (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
        _git(repo, "add", "app.py")
        _git(repo, "commit", "-m", "moved")
        with pytest.raises(WorktreeError, match="is not base_commit"):
            verify_resume_worktree(repo, sha)
