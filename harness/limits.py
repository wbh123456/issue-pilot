"""Centralized harness limits.

These are mechanical guardrails, not prompt instructions. Diagnose increments
``retry_count``; if ``retry_count >= MAX_RETRY`` the graph asks for optional
same-process feedback. With ``MAX_RETRY=2`` that is one initial execute plus
one automatic retry. ``MAX_HUMAN_RETRY=1`` allows one feedback-guided replan
after that; blank, missing, or a second request escalates to ``needs_human``.
"""

from __future__ import annotations

MAX_AGENT_STEPS = 15
MAX_RETRY = 2
MAX_HUMAN_RETRY = 1
MAX_TOOL_OUTPUT = 10_000
COMMAND_TIMEOUT = 60
AGENT_TEMPERATURE = 0
# LangGraph's default is 25. V2 + two diagnose loops + feedback + approval
# is ~20 node visits; 40 leaves headroom without changing routing.
GRAPH_RECURSION_LIMIT = 40


def truncate_output(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return (
        text[:limit]
        + f"\n\n...[truncated {omitted} chars; MAX_TOOL_OUTPUT={limit}]"
    )
