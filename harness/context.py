"""Mechanical retrieval / context budgets.

These are harness limits, not prompt instructions. Retrieval still runs on the
host (same trust class as ``grep_code``); the planner cannot ask for a larger K.
"""

from __future__ import annotations

RETRIEVE_K = 5
RRF_K = 60
MAX_CHUNK_CHARS = 4_000
MAX_PLANNER_CONTEXT_CHARS = 8_000


def truncate_chars(text: str, limit: int, *, label: str) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n\n...[truncated {omitted} chars; {label}={limit}]"
