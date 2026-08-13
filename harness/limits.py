"""Centralized harness limits.

These are mechanical guardrails, not prompt instructions. ``MAX_RETRY`` is
configuration-only in Phase 3 — the Day 5 retry/replan loop is not wired yet.
"""

from __future__ import annotations

MAX_AGENT_STEPS = 15
MAX_RETRY = 2
MAX_TOOL_OUTPUT = 10_000
COMMAND_TIMEOUT = 60


def truncate_output(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        text[:limit]
        + f"\n\n...[truncated {omitted} chars; MAX_TOOL_OUTPUT={limit}]"
    )
