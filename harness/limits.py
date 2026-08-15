"""Centralized harness limits.

These are mechanical guardrails, not prompt instructions. Diagnose increments
``retry_count``; if ``retry_count >= MAX_RETRY`` the graph escalates. With
``MAX_RETRY=2`` that is one initial execute plus one retry, then
``needs_human``.
"""

from __future__ import annotations

MAX_AGENT_STEPS = 15
MAX_RETRY = 2
MAX_TOOL_OUTPUT = 10_000
COMMAND_TIMEOUT = 60
AGENT_TEMPERATURE = 0


def truncate_output(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        text[:limit]
        + f"\n\n...[truncated {omitted} chars; MAX_TOOL_OUTPUT={limit}]"
    )
