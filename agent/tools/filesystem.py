"""Filesystem tools: list_files, read_file, edit_file."""

from __future__ import annotations

from pathlib import Path

from ._sandbox import (
    rel_to_repo,
    resolve_in_repo,
    should_skip_dir,
    truncate_output,
)


def list_files(repo_path: str | Path, path: str = ".") -> str:
    """List entries under ``path`` (relative to the benchmark repo root).

    Returns one entry per line. Directories end with ``/``.
    """
    target = resolve_in_repo(repo_path, path)
    if not target.exists():
        return f"Error: path not found: {path}"
    if not target.is_dir():
        return f"Error: not a directory: {path}"

    entries: list[str] = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.is_dir() and should_skip_dir(child.name):
            continue
        if child.name.startswith(".") and child.name not in {".gitignore", ".env.example"}:
            # Still show normal hidden config files if present; skip VCS/cache dirs above.
            if child.is_dir():
                continue
        rel = rel_to_repo(repo_path, child)
        entries.append(f"{rel}/" if child.is_dir() else rel)

    if not entries:
        return "(empty)"
    return truncate_output("\n".join(entries))


def read_file(repo_path: str | Path, path: str) -> str:
    """Read a text file relative to the benchmark repo root.

    Long outputs are truncated at ``MAX_TOOL_OUTPUT``.
    """
    if not path or not path.strip():
        return "Error: path is required"

    target = resolve_in_repo(repo_path, path)
    if not target.exists():
        return f"Error: file not found: {path}"
    if target.is_dir():
        return f"Error: path is a directory (use list_files): {path}"
    if not target.is_file():
        return f"Error: not a regular file: {path}"

    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: file is not valid UTF-8 text: {path}"
    except OSError as exc:
        return f"Error: could not read {path}: {exc}"

    return truncate_output(text)


def edit_file(
    repo_path: str | Path,
    path: str,
    content: str | None = None,
    old_str: str | None = None,
    new_str: str | None = None,
) -> str:
    """Edit a text file inside the benchmark sandbox.

    Two modes (exactly one):
      * ``content`` — write the entire file (creates parents / new files).
      * ``old_str`` / ``new_str`` — replace exactly one occurrence of ``old_str``.
    """
    if not path or not path.strip():
        return "Error: path is required"

    use_content = content is not None
    use_replace = old_str is not None
    if use_content == use_replace:
        return (
            "Error: provide exactly one of "
            "`content` (full write) or `old_str`/`new_str` (search-replace)"
        )
    if use_replace and new_str is None:
        return "Error: `new_str` is required when using search-replace"

    target = resolve_in_repo(repo_path, path)
    if target.exists() and target.is_dir():
        return f"Error: path is a directory: {path}"

    try:
        if use_content:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Re-check after mkdir that we still land inside the sandbox.
            resolve_in_repo(repo_path, path)
            target.write_text(content, encoding="utf-8")
            rel = rel_to_repo(repo_path, target)
            return f"Wrote {len(content)} chars to {rel}"

        if not target.exists():
            return f"Error: file not found: {path}"
        if not target.is_file():
            return f"Error: not a regular file: {path}"

        text = target.read_text(encoding="utf-8")
        count = text.count(old_str)
        if count == 0:
            return f"Error: old_str not found in {path}"
        if count > 1:
            return (
                f"Error: old_str found {count} times in {path}; "
                "must match exactly once"
            )
        updated = text.replace(old_str, new_str, 1)
        target.write_text(updated, encoding="utf-8")
        rel = rel_to_repo(repo_path, target)
        return f"Updated {rel} ({len(old_str)} -> {len(new_str)} chars)"
    except UnicodeDecodeError:
        return f"Error: file is not valid UTF-8 text: {path}"
    except OSError as exc:
        return f"Error: could not edit {path}: {exc}"

