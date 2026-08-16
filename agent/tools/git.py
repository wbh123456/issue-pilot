"""Git tools: git_diff via the active Docker sandbox."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from harness.limits import COMMAND_TIMEOUT
from harness.permissions import CommandPermissionError
from sandbox.runner import CommandResult, SandboxUnusableError

from ._sandbox import resolve_repo_root, truncate_output

if TYPE_CHECKING:
    from sandbox.runner import SandboxRunner

_GIT_ERROR_DETAIL_LIMIT = 180


def parse_status_paths(status: str) -> tuple[list[str], list[str]]:
    """Split ``git status --short`` into tracked changes and untracked paths."""
    changed: list[str] = []
    untracked: list[str] = []
    seen: set[str] = set()

    def _add(bucket: list[str], path: str) -> None:
        cleaned = path.strip().strip('"')
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        bucket.append(cleaned)

    for raw in (status or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("??"):
            _add(untracked, line[2:])
            continue
        rest = line[3:] if len(line) >= 4 else line.lstrip()
        if " -> " in rest:
            src, dst = rest.split(" -> ", 1)
            _add(changed, src)
            _add(changed, dst)
        else:
            _add(changed, rest)
    return changed, untracked


def format_file_lists(
    changed_files: list[str] | None = None,
    untracked_files: list[str] | None = None,
) -> str:
    """Human-readable path lists for evaluate / diagnose prompts."""
    changed = [p for p in (changed_files or []) if p]
    untracked = [p for p in (untracked_files or []) if p]
    lines = ["Changed files:"]
    if changed:
        lines.extend(f"- {path}" for path in changed)
    else:
        lines.append("- (none)")
    lines.append("Untracked files:")
    if untracked:
        lines.extend(f"- {path}" for path in untracked)
    else:
        lines.append("- (none)")
    return "\n".join(lines)


@dataclass
class WorktreeDiff:
    """Structured working-tree inspection. ``valid`` is a non-empty clean patch."""

    status: str = ""
    diff: str = ""
    error: str | None = None
    timed_out: bool = False
    changed_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        if self.error is not None or self.timed_out:
            return False
        return bool(self.status.strip() or self.diff.strip())

    def to_dict(self) -> dict:
        return asdict(self)

    def format(self) -> str:
        if self.error:
            return self.error
        if not self.status.strip() and not self.diff.strip():
            return "(no changes)"
        parts: list[str] = []
        if self.status.strip():
            parts.append("--- status ---\n" + self.status.strip())
        if self.diff.strip():
            parts.append("--- diff ---\n" + self.diff.strip())
        else:
            parts.append(
                "--- diff ---\n(no tracked-file diff; see status for untracked)"
            )
        return truncate_output("\n".join(parts))


def _short_git_error(label: str, result: CommandResult) -> str:
    """One-line error; never dump git's usage/help into the agent context."""
    raw = (result.stderr or result.stdout or "").strip() or "unknown"
    first = raw.splitlines()[0].strip()
    if len(first) > _GIT_ERROR_DETAIL_LIMIT:
        first = first[:_GIT_ERROR_DETAIL_LIMIT] + "…"
    return f"Error: {label} (exit {result.exit_code}): {first}"


def inspect_worktree(repo_path: str | Path, *, sandbox: SandboxRunner) -> WorktreeDiff:
    """Inspect the working tree against HEAD inside the sandbox."""
    if sandbox is None:
        raise RuntimeError(
            "inspect_worktree requires an active SandboxRunner; host execution is not allowed"
        )

    resolve_repo_root(repo_path)
    try:
        diff = sandbox.run(["git", "diff", "HEAD"])
        status = sandbox.run(["git", "status", "--short"])
    except CommandPermissionError as exc:
        return WorktreeDiff(error=f"Error: command not allowed: {exc}")
    except SandboxUnusableError as exc:
        return WorktreeDiff(error=f"Error: sandbox unusable: {exc}")

    if diff.timed_out or status.timed_out:
        return WorktreeDiff(
            timed_out=True,
            error=f"Error: git timed out after {COMMAND_TIMEOUT}s",
        )

    if diff.exit_code not in (0, 1):
        return WorktreeDiff(error=_short_git_error("git diff failed", diff))
    if status.exit_code != 0:
        return WorktreeDiff(error=_short_git_error("git status failed", status))

    status_text = (status.stdout or "").rstrip()
    changed_files, untracked_files = parse_status_paths(status_text)
    return WorktreeDiff(
        status=status_text,
        diff=(diff.stdout or "").rstrip(),
        changed_files=changed_files,
        untracked_files=untracked_files,
    )


def git_diff(repo_path: str | Path, *, sandbox: SandboxRunner) -> str:
    """Return the working-tree diff against HEAD for the benchmark repo.

    Runs fixed ``git status --short`` and ``git diff HEAD`` inside the sandbox.
    Includes short status so newly created (untracked) files are visible.
    """
    if sandbox is None:
        raise RuntimeError(
            "git_diff requires an active SandboxRunner; host execution is not allowed"
        )
    return inspect_worktree(repo_path, sandbox=sandbox).format()
