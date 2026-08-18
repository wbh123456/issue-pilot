"""Analyze node: one no-tool LLM call for problem + hypothesis."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from harness.limits import AGENT_TEMPERATURE

from ._runtime import get_reporter, merge_telemetry, require_config, stage_usage, traced

ANALYZE_SYSTEM = """
You are analyzing a software bug before any code changes.

Write a short analysis that covers:
1. the problem in plain terms
2. an initial hypothesis about the root cause

Do not propose file edits yet. Do not call tools. Plain text only.
""".strip()


def analyze_issue(state: AgentState, config: RunnableConfig) -> dict:
    cfg = require_config(config, "client", "model")
    client = cfg["client"]
    model = cfg["model"]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ANALYZE_SYSTEM},
            {
                "role": "user",
                "content": f"Analyze this issue:\n\n{state['issue']}",
            },
        ],
        temperature=AGENT_TEMPERATURE,
    )
    analysis = (response.choices[0].message.content or "").strip()
    if not analysis:
        raise RuntimeError("Analyze node returned empty content")

    reporter = get_reporter(config)
    reporter.stage("analyze")
    reporter.note(analysis)

    return traced(
        state,
        {
            "analysis": analysis,
            "status": "analyzed",
            "telemetry": merge_telemetry(state, **stage_usage("analyze", response)),
        },
        node="analyze",
    )
