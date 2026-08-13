"""Shared path sandbox helpers for agent tools.

Host-side file APIs (list/read/edit) may only touch the benchmark repo root.
They never receive the harness repo, ``.env``, Docker socket, or other mounts.
Command execution is a separate boundary (see ``harness.permissions`` + Docker).
"""

from __future__ import annotations

from pathlib import Path

from harness.limits import MAX_TOOL_OUTPUT, truncate_output

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
    "_gold",
    "_app_bak",
}

__all__ = [
    "MAX_TOOL_OUTPUT",
    "rel_to_repo",
    "resolve_in_repo",
    "resolve_repo_root",
    "should_skip_dir",
    "truncate_output",
]


def resolve_repo_root(repo_path: str | Path) -> Path:
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repo_path is not a directory: {root}")
    return root


def resolve_in_repo(repo_path: str | Path, rel_path: str = ".") -> Path:
    """Resolve ``rel_path`` under ``repo_path``; reject escapes outside the repo.

    Symlinks are followed via ``Path.resolve()``. A link that points outside
    the benchmark root raises ``PermissionError``. Absolute inputs that resolve
    outside the root are likewise rejected.
    """
    root = resolve_repo_root(repo_path)
    candidate = rel_path.strip() or "."
    # resolve() follows symlinks; relative_to rejects anything outside root.
    target = (root / candidate).resolve()
    try:
        rel = target.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"path escapes sandbox root: {rel_path!r} (repo={root})"
        ) from exc
    if any(part == "_gold" for part in rel.parts):
        raise PermissionError(
            f"path is not visible to agent tools: {rel_path!r}"
        )
    return target


def rel_to_repo(repo_path: str | Path, absolute: Path) -> str:
    root = resolve_repo_root(repo_path)
    return absolute.resolve().relative_to(root).as_posix()


def should_skip_dir(name: str) -> bool:
    return name in _SKIP_DIR_NAMES
