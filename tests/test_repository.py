"""Host git helpers used by evaluator reset and resume worktree checks."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eval.repository import (
    WorktreeError,
    capture_patch_diff,
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


class TestCapturePatchDiff:
    def test_missing_repo_is_empty(self, tmp_path: Path) -> None:
        assert capture_patch_diff(tmp_path / "nope") == ""

    def test_clean_worktree_is_empty(self, tmp_path: Path) -> None:
        repo, _sha = _init_repo(tmp_path)
        assert capture_patch_diff(repo) == ""

    def test_includes_tracked_edit(self, tmp_path: Path) -> None:
        repo, _sha = _init_repo(tmp_path)
        (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
        diff = capture_patch_diff(repo)
        assert "app.py" in diff
        assert "+x = 2" in diff

    def test_includes_untracked_and_skips_gold_staging(self, tmp_path: Path) -> None:
        repo, _sha = _init_repo(tmp_path)
        (repo / "new_mod.py").write_text("y = 1\n", encoding="utf-8")
        gold = repo / "tests" / "_gold"
        gold.mkdir(parents=True)
        (gold / "secret.py").write_text("assert False\n", encoding="utf-8")
        diff = capture_patch_diff(repo)
        assert "new_mod.py" in diff
        assert "+y = 1" in diff
        assert "secret.py" not in diff
        assert "_gold" not in diff
