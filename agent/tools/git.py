"""Git tools: git_diff via the active Docker sandbox."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from harness.limits import COMMAND_TIMEOUT
from harness.permissions import CommandPermissionError
from sandbox.runner import SandboxUnusableError

from ._sandbox import resolve_repo_root, truncate_output

if TYPE_CHECKING:
    from sandbox.runner import SandboxRunner


def git_diff(repo_path: str | Path, *, sandbox: SandboxRunner) -> str:
    """Return the working-tree diff against HEAD for the benchmark repo.

    Runs fixed ``git status --short`` and ``git diff HEAD`` inside the sandbox.
    Includes short status so newly created (untracked) files are visible.
    """
    if sandbox is None:
        raise RuntimeError(
            "git_diff requires an active SandboxRunner; host execution is not allowed"
        )

    resolve_repo_root(repo_path)
    try:
        diff = sandbox.run(["git", "diff", "HEAD"])
        status = sandbox.run(["git", "status", "--short"])
    except CommandPermissionError as exc:
        return f"Error: command not allowed: {exc}"
    except SandboxUnusableError as exc:
        return f"Error: sandbox unusable: {exc}"

    if diff.timed_out or status.timed_out:
        return f"Error: git timed out after {COMMAND_TIMEOUT}s"

    if diff.exit_code not in (0, 1):
        err = (diff.stderr or diff.stdout or "").strip()
        return f"Error: git diff failed (exit {diff.exit_code}): {err or 'unknown'}"
    if status.exit_code != 0:
        err = (status.stderr or status.stdout or "").strip()
        return f"Error: git status failed (exit {status.exit_code}): {err or 'unknown'}"

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
