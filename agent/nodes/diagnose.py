"""Diagnose node: one-shot failure analysis; does not re-execute."""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState

from ._runtime import merge_telemetry, require_config, stage_usage

DIAGNOSE_SYSTEM = """
You are diagnosing why a coding-agent attempt failed verification.

Given the issue, plan, git diff, and failing test output, explain:
1. the most likely cause of the failure
2. what was wrong or incomplete in the attempt
3. what should be tried next

Do not call tools. Do not edit code. Plain text only.
This diagnosis will be recorded; it will not be executed again in this phase.
""".strip()


def diagnose_failure(state: AgentState, config: RunnableConfig) -> dict:
    cfg = require_config(config, "client", "model")
    client = cfg["client"]
    model = cfg["model"]
    test_result = state.get("test_result") or {}

    plan_text = json.dumps(state.get("plan") or {}, ensure_ascii=False, indent=2)
    test_text = json.dumps(
        {
            "command": test_result.get("command"),
            "exit_code": test_result.get("exit_code"),
            "passed": test_result.get("passed"),
            "output": test_result.get("output"),
        },
        ensure_ascii=False,
        indent=2,
    )
    user_content = (
        f"Issue:\n{state['issue']}\n\n"
        f"Plan:\n{plan_text}\n\n"
        f"Test result:\n{test_text}\n\n"
        f"Git diff:\n{test_result.get('git_diff') or '(none)'}\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DIAGNOSE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    diagnosis = (response.choices[0].message.content or "").strip()
    if not diagnosis:
        raise RuntimeError("Diagnose node returned empty content")

    return {
        "diagnosis": diagnosis,
        "status": "failed",
        "telemetry": merge_telemetry(state, **stage_usage("diagnose", response)),
    }
