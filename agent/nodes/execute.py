"""Execute node: reuse the V0 ReAct tool loop with workflow context."""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig

from agent.loop import run_agent
from agent.state import AgentState
from agent.tools.schema import TOOLS, V2_TOOLS
from harness.limits import MAX_AGENT_STEPS
from retrieval.dense import Embedder, make_embedder

from ._runtime import get_reporter, merge_telemetry, require_config

_EXECUTOR_GUARDRAIL = (
    "Ignore plan paths that are not present in the repository. "
    "Prefer the existing Python/FastAPI layout under app/ and tests/."
)


def _workflow_context(state: AgentState) -> str:
    """Pass the structured plan; V2 also includes retrieved snippets."""
    plan = state.get("plan") or {}
    payload = {
        "plan": plan,
        "guardrail": _EXECUTOR_GUARDRAIL,
    }
    snippets = (state.get("retrieved_context") or "").strip()
    if snippets:
        payload["retrieved"] = snippets
    files = [p for p in (state.get("relevant_files") or []) if p]
    if files:
        payload["relevant_files"] = files
    diagnosis = (state.get("diagnosis") or "").strip()
    if diagnosis:
        payload["diagnosis"] = diagnosis
        payload["retry_count"] = int(state.get("retry_count") or 0)
    structured = state.get("structured_diagnosis") or {}
    if structured:
        payload["structured_diagnosis"] = structured
    history = list(state.get("attempt_history") or [])
    if history:
        payload["attempt_history"] = history
    feedback = (state.get("human_feedback") or "").strip()
    if feedback:
        payload["human_feedback"] = feedback
        payload["human_retry_count"] = int(state.get("human_retry_count") or 0)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _embedder_from_execute_config(cfg: dict) -> Embedder | None:
    if cfg.get("embedder") is not None:
        return cfg["embedder"]
    name = cfg.get("embedder_name")
    if not name:
        return None
    return make_embedder(str(name))


def execute_plan(state: AgentState, config: RunnableConfig) -> dict:
    cfg = require_config(config, "client", "model", "repo_path", "test_command")
    enable_search = bool(cfg.get("enable_search_code"))
    embedder = _embedder_from_execute_config(cfg)
    reporter = get_reporter(config)
    retry = int(state.get("retry_count") or 0)
    if retry:
        reporter.stage("execute", f"retry {retry}")
    else:
        reporter.stage("execute")
    result = run_agent(
        client=cfg["client"],
        issue=state["issue"],
        repo_path=cfg["repo_path"],
        test_command=cfg["test_command"],
        model=cfg["model"],
        max_steps=int(cfg.get("max_steps", MAX_AGENT_STEPS)),
        workflow_context=_workflow_context(state),
        sandbox=cfg.get("sandbox"),
        tools=V2_TOOLS if enable_search else TOOLS,
        search_code_enabled=enable_search,
        embedder=embedder,
        progress=cfg.get("progress"),
    )

    prompt_tokens = int(result.get("prompt_tokens", 0) or 0)
    completion_tokens = int(result.get("completion_tokens", 0) or 0)
    exec_steps = int(result.get("steps", 0) or 0)
    search_calls = sum(
        1
        for event in (result.get("trajectory") or [])
        if (event or {}).get("tool") == "search_code"
    )

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
        retrieval_calls=search_calls,
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
