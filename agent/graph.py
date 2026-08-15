"""Compile and run the V1 / V2 LangGraph Plan-Execute workflow."""

from __future__ import annotations

import time
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    analyze_issue,
    diagnose_failure,
    deterministic_verify,
    execute_plan,
    structured_plan,
)
from agent.state import AgentState, initial_state
from harness.limits import MAX_AGENT_STEPS


def mark_success(state: AgentState) -> dict:
    """Terminal PASS node — status only; no LLM call."""
    return {"status": "success"}


def route_after_verify(state: AgentState) -> Literal["mark_success", "diagnose"]:
    """Route strictly from the deterministic process exit code."""
    test_result = state.get("test_result") or {}
    if int(test_result.get("exit_code", 1)) == 0:
        return "mark_success"
    return "diagnose"


def build_graph(*, include_retrieve: bool = False):
    """Build analyze → [retrieve] → plan → execute → verify → (PASS | diagnose).

    Default (``include_retrieve=False``) is the V1 graph. V2 inserts a
    deterministic retrieve node between analyze and plan.
    """
    graph = StateGraph(AgentState)
    graph.add_node("analyze", analyze_issue)
    graph.add_node("plan", structured_plan)
    graph.add_node("execute", execute_plan)
    graph.add_node("verify", deterministic_verify)
    graph.add_node("diagnose", diagnose_failure)
    graph.add_node("mark_success", mark_success)

    graph.add_edge(START, "analyze")
    if include_retrieve:
        from agent.nodes.retrieve import retrieve_context

        graph.add_node("retrieve", retrieve_context)
        graph.add_edge("analyze", "retrieve")
        graph.add_edge("retrieve", "plan")
    else:
        graph.add_edge("analyze", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "mark_success": "mark_success",
            "diagnose": "diagnose",
        },
    )
    graph.add_edge("mark_success", END)
    graph.add_edge("diagnose", END)
    return graph.compile()


_GRAPH = None
_V2_GRAPH = None


def get_graph():
    """Return the V1 compiled graph (process-wide lazy singleton)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def get_v2_graph():
    """Return the V2 compiled graph (analyze → retrieve → plan). Separate singleton."""
    global _V2_GRAPH
    if _V2_GRAPH is None:
        _V2_GRAPH = build_graph(include_retrieve=True)
    return _V2_GRAPH


def adapt_result(state: AgentState) -> dict[str, Any]:
    """Map final graph state to the common evaluator / V0 result keys."""
    telemetry = state.get("telemetry") or {}
    test_result = state.get("test_result") or {}
    status = state.get("status") or ""
    passed = int(test_result.get("exit_code", 1)) == 0

    if status == "success" or passed:
        termination = "completed"
    elif status == "failed":
        termination = "failed"
    else:
        termination = telemetry.get("termination") or status or "unknown"

    return {
        "final_answer": telemetry.get("final_answer", ""),
        "termination": termination,
        "steps": telemetry.get("steps", 0),
        "tool_call_count": telemetry.get("tool_call_count", 0),
        "file_reads": telemetry.get("file_reads", 0),
        "prompt_tokens": telemetry.get("prompt_tokens", 0),
        "completion_tokens": telemetry.get("completion_tokens", 0),
        "tokens": telemetry.get("tokens", 0),
        "latency": telemetry.get("latency", 0.0),
        "trajectory": telemetry.get("trajectory", []),
        "messages": telemetry.get("messages", []),
        "stage_tokens": telemetry.get("stage_tokens", {}),
        # V1 workflow fields (runner can persist these later)
        "analysis": state.get("analysis", ""),
        "plan": state.get("plan", {}),
        "diagnosis": state.get("diagnosis", ""),
        "test_result": test_result,
        "status": status,
        "llm_calls": telemetry.get("llm_calls", 0),
        "workflow_passed": passed,
        "relevant_files": state.get("relevant_files") or [],
        "retrieval_calls": telemetry.get("retrieval_calls", 0),
    }


def run_workflow(
    *,
    client,
    issue: str,
    repo_path: str,
    test_command: str,
    model: str = "deepseek-v4-flash",
    max_steps: int = MAX_AGENT_STEPS,
    graph=None,
    sandbox=None,
    enable_search_code: bool = False,
    embedder_name: str = "hashing",
) -> dict[str, Any]:
    """Invoke the compiled graph and return evaluator-compatible result keys.

    Defaults to the V1 singleton. Pass ``graph=get_v2_graph()`` and
    ``enable_search_code=True`` for V2.
    Runtime objects (LLM client, SandboxRunner) stay in ``configurable``,
    not in serializable ``AgentState``.
    """
    compiled = graph or get_graph()
    started_at = time.perf_counter()
    final_state = compiled.invoke(
        initial_state(issue),
        config={
            "configurable": {
                "client": client,
                "model": model,
                "repo_path": repo_path,
                "test_command": test_command,
                "max_steps": max_steps,
                "sandbox": sandbox,
                "enable_search_code": enable_search_code,
                "embedder_name": embedder_name,
            }
        },
    )
    result = adapt_result(final_state)
    # Wall-clock for the full V1 workflow (analyze/plan/execute/verify/diagnose).
    result["latency"] = time.perf_counter() - started_at
    return result
