"""Diagnose node: one-shot failure analysis; graph may re-execute after this."""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from harness.limits import AGENT_TEMPERATURE, MAX_RETRY
from harness.progress import preview

from ._runtime import get_reporter, merge_telemetry, require_config, stage_usage

DIAGNOSE_SYSTEM = """
You are diagnosing why a coding-agent attempt failed verification.

Given the issue, plan, git diff, and failing test output, explain:
1. the most likely cause of the failure
2. what was wrong or incomplete in the attempt
3. what should be tried next

Do not call tools. Do not edit code. Plain text only.
If retries remain, the executor will run again with this diagnosis.
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
        temperature=AGENT_TEMPERATURE,
    )
    diagnosis = (response.choices[0].message.content or "").strip()
    if not diagnosis:
        raise RuntimeError("Diagnose node returned empty content")

    retry_count = int(state.get("retry_count") or 0) + 1
    status = "failed" if retry_count >= MAX_RETRY else "retrying"
    get_reporter(config).stage("diagnose", preview(diagnosis))
    return {
        "diagnosis": diagnosis,
        "retry_count": retry_count,
        "status": status,
        "telemetry": merge_telemetry(state, **stage_usage("diagnose", response)),
    }
