"""Code search tool: grep_code (pure Python; no host subprocess)."""

from __future__ import annotations

from pathlib import Path

from harness.context import RETRIEVE_K
from retrieval.dense import Embedder, HashingEmbedder
from retrieval.indexer import build_index

from ._sandbox import (
    MAX_TOOL_OUTPUT,
    rel_to_repo,
    resolve_repo_root,
    should_skip_dir,
    truncate_output,
)


def grep_code(repo_path: str | Path, query: str) -> str:
    """Search the benchmark repo for ``query`` using a path-jailed Python walk."""
    if not query:
        return "Error: query is required"

    root = resolve_repo_root(repo_path)
    return truncate_output(_grep_with_python(root, query))


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


def search_code(
    repo_path: str | Path,
    query: str,
    *,
    embedder: Embedder | None = None,
    k: int = RETRIEVE_K,
) -> str:
    """Hybrid symbol search. Host-side, same trust class as ``grep_code``."""
    if not query or not str(query).strip():
        return "Error: query is required"

    resolve_repo_root(repo_path)
    index = build_index(repo_path, embedder=embedder or HashingEmbedder())
    chunks = index.search_hybrid(str(query).strip(), k=k)
    if not chunks:
        return "(no matches)"

    parts: list[str] = []
    for chunk in chunks:
        header = (
            f"{chunk.path}:{chunk.start_line}-{chunk.end_line}:"
            f"{chunk.symbol} ({chunk.type})"
        )
        parts.append(f"{header}\n{chunk.text}")
    return truncate_output("\n\n".join(parts))
