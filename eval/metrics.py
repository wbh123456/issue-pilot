"""File-level retrieval metrics. Independent of the agent / LLM."""

from __future__ import annotations

from collections.abc import Sequence


def normalize_path(path: str) -> str:
    """POSIX-ish relative path for comparing ``expected_files``."""
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def unique_paths(paths: Sequence[str], *, k: int | None = None) -> list[str]:
    """Collapse a ranked path list to unique files, preserving order."""
    seen: list[str] = []
    known: set[str] = set()
    for raw in paths:
        path = normalize_path(raw)
        if not path or path in known:
            continue
        known.add(path)
        seen.append(path)
        if k is not None and len(seen) >= k:
            break
    return seen


def recall_at_k(
    retrieved_files: Sequence[str],
    expected_files: Sequence[str],
    k: int = 5,
) -> float:
    """``|unique(retrieved)[:k] ∩ expected| / |expected|``.

    Chunk hits should be collapsed to paths before (or via) this helper.
    Empty ``expected_files`` is vacuously 1.0.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    expected = {normalize_path(p) for p in expected_files if p and str(p).strip()}
    if not expected:
        return 1.0
    top = unique_paths(retrieved_files, k=k)
    hits = sum(1 for path in top if path in expected)
    return hits / len(expected)
