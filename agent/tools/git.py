"""Git tools: git_diff."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ._sandbox import resolve_repo_root, truncate_output

_GIT_CANDIDATES = (
    "git",
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files (x86)\Git\cmd\git.exe",
)


def _find_git() -> str | None:
    found = shutil.which("git")
    if found:
        return found
    for candidate in _GIT_CANDIDATES[1:]:
        if Path(candidate).is_file():
            return candidate
    return None


def _run_git(git: str, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [git, "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )


def git_diff(repo_path: str | Path) -> str:
    """Return the working-tree diff against HEAD for the benchmark repo.

    Includes short status so newly created (untracked) files are visible.
    """
    root = resolve_repo_root(repo_path)
    git = _find_git()
    if git is None:
        return "Error: git executable not found"

    try:
        diff = _run_git(git, root, "diff", "HEAD")
        status = _run_git(git, root, "status", "--short")
    except subprocess.TimeoutExpired:
        return "Error: git timed out after 30s"
    except OSError as exc:
        return f"Error: failed to run git: {exc}"

    if diff.returncode not in (0, 1):
        err = (diff.stderr or diff.stdout or "").strip()
        return f"Error: git diff failed (exit {diff.returncode}): {err or 'unknown'}"
    if status.returncode != 0:
        err = (status.stderr or status.stdout or "").strip()
        return f"Error: git status failed (exit {status.returncode}): {err or 'unknown'}"

    diff_text = (diff.stdout or "").rstrip()
    status_text = (status.stdout or "").rstrip()

    if not diff_text and not status_text:
        return "(no changes)"

    parts: list[str] = []
    if status_text:
        parts.append("--- status ---\n" + status_text)
    if diff_text:
        parts.append("--- diff ---\n" + diff_text)
    else:
        parts.append("--- diff ---\n(no tracked-file diff; see status for untracked)")
    return truncate_output("\n".join(parts))
