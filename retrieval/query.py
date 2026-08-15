"""Shared retrieval query construction for offline eval and the V2 retrieve node."""

from __future__ import annotations

from typing import Literal

QueryMode = Literal["issue", "issue+analysis"]
QUERY_MODES: tuple[QueryMode, ...] = ("issue", "issue+analysis")
DEFAULT_QUERY_MODE: QueryMode = "issue"


def normalize_query_mode(mode: str | None) -> QueryMode:
    key = (mode or DEFAULT_QUERY_MODE).strip().lower()
    if key in {"issue", "raw", "issue-only"}:
        return "issue"
    if key in {"issue+analysis", "issue-analysis", "analysis"}:
        return "issue+analysis"
    raise ValueError(
        f"unknown retrieve query mode {mode!r}; use 'issue' or 'issue+analysis'"
    )


def build_retrieval_query(
    issue: str,
    analysis: str = "",
    *,
    mode: str = DEFAULT_QUERY_MODE,
) -> str:
    """Return the search string for hybrid / BM25 / dense / grep ranking.

    Default ``issue`` matches the offline Recall@K CLI. ``issue+analysis``
    appends the analyzer output when both strings are non-empty.
    """
    issue_text = (issue or "").strip()
    if normalize_query_mode(mode) == "issue":
        return issue_text
    analysis_text = (analysis or "").strip()
    if issue_text and analysis_text:
        return f"{issue_text}\n{analysis_text}"
    return issue_text or analysis_text
