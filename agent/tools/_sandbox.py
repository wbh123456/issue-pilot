"""Shared path sandbox helpers for agent tools."""

from __future__ import annotations

from pathlib import Path

# Day 3 will enforce this more strictly; Day 1 truncates early for headroom.
MAX_TOOL_OUTPUT = 10_000

_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
}


def resolve_repo_root(repo_path: str | Path) -> Path:
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repo_path is not a directory: {root}")
    return root


def resolve_in_repo(repo_path: str | Path, rel_path: str = ".") -> Path:
    """Resolve ``rel_path`` under ``repo_path``; reject escapes outside the repo."""
    root = resolve_repo_root(repo_path)
    candidate = rel_path.strip() or "."
    # Absolute inputs are treated as relative to the sandbox root by name only.
    target = (root / candidate).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"path escapes sandbox root: {rel_path!r} (repo={root})"
        ) from exc
    return target


def rel_to_repo(repo_path: str | Path, absolute: Path) -> str:
    root = resolve_repo_root(repo_path)
    return absolute.resolve().relative_to(root).as_posix()


def truncate_output(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        text[:limit]
        + f"\n\n...[truncated {omitted} chars; MAX_TOOL_OUTPUT={limit}]"
    )


def should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIR_NAMES
