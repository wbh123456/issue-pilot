"""Trusted host-only repository orchestration.

These commands are evaluator/admin operations with fixed argv. They are not
agent tools and must never take model-controlled arguments.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

GOLD_STAGING_DIRNAME = "_gold"

_GIT_CANDIDATES = (
    "git",
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
)


def find_host_git() -> str | None:
    """Locate a host git executable for evaluator reset/clean only."""
    found = shutil.which("git")
    if found:
        return found
    for candidate in _GIT_CANDIDATES[1:]:
        if Path(candidate).is_file():
            return candidate
    return None


def reset_repo(repo_path: Path, base_commit: str) -> None:
    """Hard-reset the benchmark repo to ``base_commit`` and clean extras."""
    git = find_host_git()
    if git is None:
        raise RuntimeError("git executable not found")

    commands = [
        [git, "-C", str(repo_path), "reset", "--hard", base_commit],
        [git, "-C", str(repo_path), "clean", "-fd"],
    ]
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"git command failed ({' '.join(cmd)}): {err or 'unknown error'}"
            )

    leftover = repo_path / "tests" / GOLD_STAGING_DIRNAME
    if leftover.exists():
        shutil.rmtree(leftover, ignore_errors=True)


class WorktreeError(RuntimeError):
    """Resume refused because the benchmark worktree is not the paused patch."""


def _run_host_git(repo_path: Path, *args: str) -> subprocess.CompletedProcess:
    git = find_host_git()
    if git is None:
        raise RuntimeError("git executable not found")
    return subprocess.run(
        [git, "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def verify_resume_worktree(repo_path: Path, base_commit: str) -> None:
    """Require HEAD at ``base_commit`` and an uncommitted patch.

    Resume must never call ``reset_repo``: the agent patch lives in the
    bind-mounted worktree and would be destroyed by a hard reset.
    """
    head = _run_host_git(repo_path, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        err = (head.stderr or head.stdout or "").strip()
        raise WorktreeError(f"cannot read HEAD: {err or 'unknown error'}")
    expected = _run_host_git(repo_path, "rev-parse", "--verify", f"{base_commit}^{{commit}}")
    if expected.returncode != 0:
        raise WorktreeError(f"base_commit is not a revision: {base_commit}")
    head_sha = (head.stdout or "").strip()
    base_sha = (expected.stdout or "").strip()
    if head_sha != base_sha:
        raise WorktreeError(
            f"resume worktree HEAD {head_sha} is not base_commit {base_sha}"
        )
    status = _run_host_git(repo_path, "status", "--porcelain")
    if status.returncode != 0:
        err = (status.stderr or status.stdout or "").strip()
        raise WorktreeError(f"cannot read worktree status: {err or 'unknown error'}")
    if not (status.stdout or "").strip():
        raise WorktreeError("resume requires the agent patch; worktree is clean")


def git_sha(repo_path: Path) -> str | None:
    """Return HEAD SHA for ``repo_path``, or None if git is unavailable."""
    git = find_host_git()
    if git is None:
        return None
    proc = subprocess.run(
        [git, "-C", str(repo_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return None
    sha = (proc.stdout or "").strip()
    return sha or None
