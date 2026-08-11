"""Code search tool: grep_code (ripgrep preferred, Python fallback)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ._sandbox import (
    MAX_TOOL_OUTPUT,
    rel_to_repo,
    resolve_repo_root,
    should_skip_dir,
    truncate_output,
)


def grep_code(repo_path: str | Path, query: str) -> str:
    """Search the benchmark repo for ``query``.

    Prefers ``rg`` when available; otherwise walks source files in Python.
    """
    if not query:
        return "Error: query is required"

    root = resolve_repo_root(repo_path)
    rg = shutil.which("rg")
    if rg:
        return truncate_output(_grep_with_rg(rg, root, query))
    return truncate_output(_grep_with_python(root, query))


def _grep_with_rg(rg: str, root: Path, query: str) -> str:
    cmd = [
        rg,
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color",
        "never",
        "--glob",
        "!.git/**",
        "--glob",
        "!**/__pycache__/**",
        "--glob",
        "!**/.pytest_cache/**",
        "--glob",
        "!**/.venv/**",
        "--glob",
        "!**/venv/**",
        "--max-columns",
        "200",
        "--max-columns-preview",
        query,
        str(root),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "Error: rg timed out after 30s"
    except OSError as exc:
        return f"Error: failed to run rg: {exc}"

    # rg exit 0 = matches, 1 = no matches, 2+ = error
    if proc.returncode == 0:
        return _relativize_rg_output(root, proc.stdout) or "(no matches)"
    if proc.returncode == 1:
        return "(no matches)"
    err = (proc.stderr or proc.stdout or "").strip()
    return f"Error: rg failed (exit {proc.returncode}): {err or 'unknown error'}"


def _relativize_rg_output(root: Path, stdout: str) -> str:
    lines: list[str] = []
    root_prefix = str(root.resolve())
    for line in stdout.splitlines():
        if line.startswith(root_prefix):
            rest = line[len(root_prefix) :].lstrip("\\/")
            lines.append(rest.replace("\\", "/"))
        else:
            lines.append(line)
    return "\n".join(lines)


def _grep_with_python(root: Path, query: str) -> str:
    matches: list[str] = []
    text_suffixes = {
        ".py",
        ".txt",
        ".md",
        ".json",
        ".toml",
        ".yml",
        ".yaml",
        ".ini",
        ".cfg",
    }

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip ignored directories anywhere in the path.
        if any(should_skip_dir(part) for part in path.parts):
            continue
        if path.suffix.lower() not in text_suffixes and path.name not in {
            "Dockerfile",
            "Makefile",
            ".gitignore",
        }:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = rel_to_repo(root, path)
        for i, line in enumerate(content.splitlines(), start=1):
            if query in line:
                matches.append(f"{rel}:{i}:{line[:200]}")
                if sum(len(m) + 1 for m in matches) >= MAX_TOOL_OUTPUT:
                    return "\n".join(matches)

    return "\n".join(matches) if matches else "(no matches)"
