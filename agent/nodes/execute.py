"""Execute node: reuse the V0 ReAct tool loop with workflow context."""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig

from agent.loop import run_agent
from agent.state import AgentState
from harness.limits import MAX_AGENT_STEPS

from ._runtime import merge_telemetry, require_config

_EXECUTOR_GUARDRAIL = (
    "Ignore plan paths that are not present in the repository. "
    "Prefer the existing Python/FastAPI layout under app/ and tests/."
)


def _workflow_context(state: AgentState) -> str:
    """Pass only the structured plan (no duplicated analysis prose)."""
    plan = state.get("plan") or {}
    payload = {
        "plan": plan,
        "guardrail": _EXECUTOR_GUARDRAIL,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def execute_plan(state: AgentState, config: RunnableConfig) -> dict:
    cfg = require_config(config, "client", "model", "repo_path", "test_command")
    result = run_agent(
        client=cfg["client"],
        issue=state["issue"],
        repo_path=cfg["repo_path"],
        test_command=cfg["test_command"],
        model=cfg["model"],
        max_steps=int(cfg.get("max_steps", MAX_AGENT_STEPS)),
        workflow_context=_workflow_context(state),
        sandbox=cfg.get("sandbox"),
    )

    prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
    completion_tokens = int(result.get("completion_tokens", 0) or 0)
    exec_steps = int(result.get("steps", 0) or 0)

    # Fold executor outputs into telemetry (execution_result is optional/omitted).
    telemetry = merge_telemetry(
        state,
        tool_call_count=result.get("tool_call_count", 0),
        file_reads=result.get("file_reads", 0),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        steps=exec_steps,
        latency=result.get("latency", 0.0),
        llm_calls=exec_steps,
        final_answer=result.get("final_answer", ""),
        termination=result.get("termination", ""),
        trajectory=result.get("trajectory", []),
        messages=result.get("messages", []),
        stage_tokens={
            "execute": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "llm_calls": exec_steps,
            }
        },
    )

    return {
        "status": "executed",
        "telemetry": telemetry,
    }
