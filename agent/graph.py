"""Compile and run the V1 / V2 LangGraph Plan-Execute workflow."""

from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    analyze_issue,
    collect_feedback,
    diagnose_failure,
    deterministic_verify,
    evaluate_patch,
    execute_plan,
    structured_plan,
)
from agent.nodes._runtime import get_reporter
from agent.state import (
    AgentState,
    EvaluationValidationError,
    initial_state,
    patch_evaluation_passed,
)
from harness.limits import MAX_AGENT_STEPS, MAX_RETRY
from harness.progress import ProgressReporter


def _layer2_passed(raw: object) -> bool:
    if not raw:
        return False
    try:
        return patch_evaluation_passed(raw)
    except EvaluationValidationError:
        return False


def mark_success(state: AgentState, config: RunnableConfig) -> dict:
    """Terminal PASS node — status only; no LLM call."""
    get_reporter(config).stage("success")
    return {"status": "success"}


def mark_needs_human(state: AgentState, config: RunnableConfig) -> dict:
    """Terminal escalation after the retry budget is exhausted."""
    get_reporter(config).stage("needs_human")
    return {"status": "needs_human"}


def route_after_verify(state: AgentState) -> Literal["evaluate", "diagnose"]:
    """Layer 1 pass goes to the evaluator; fail goes to diagnose (fail-closed)."""
    test_result = state.get("test_result") or {}
    if test_result.get("deterministic_pass") is True:
        return "evaluate"
    return "diagnose"


def route_after_evaluate(state: AgentState) -> Literal["mark_success", "diagnose"]:
    """Mechanical Layer 2 pass only; never promote a Layer 1 failure."""
    test_result = state.get("test_result") or {}
    if test_result.get("deterministic_pass") is not True:
        return "diagnose"
    if _layer2_passed(state.get("patch_evaluation") or {}):
        return "mark_success"
    return "diagnose"


def route_after_diagnose(state: AgentState) -> Literal["plan", "feedback"]:
    """Replan while ``retry_count < MAX_RETRY``; otherwise ask for feedback."""
    if int(state.get("retry_count") or 0) >= MAX_RETRY:
        return "feedback"
    return "plan"


def route_after_feedback(state: AgentState) -> Literal["plan", "mark_needs_human"]:
    """One accepted feedback replan; skip/decline/refuse escalate."""
    if state.get("status") == "feedback_retry":
        return "plan"
    return "mark_needs_human"


def build_graph(*, include_retrieve: bool = False):
    """Build analyze → [retrieve] → plan → execute → verify → evaluate | diagnose.

    Default (``include_retrieve=False``) is the V1 graph. V2 inserts a
    deterministic retrieve node between analyze and plan. After the automatic
    retry budget, ``feedback`` may allow one same-process replan; otherwise
    the graph ends at ``mark_needs_human``.
    """
    graph = StateGraph(AgentState)
    graph.add_node("analyze", analyze_issue)
    graph.add_node("plan", structured_plan)
    graph.add_node("execute", execute_plan)
    graph.add_node("verify", deterministic_verify)
    graph.add_node("evaluate", evaluate_patch)
    graph.add_node("diagnose", diagnose_failure)
    graph.add_node("feedback", collect_feedback)
    graph.add_node("mark_success", mark_success)
    graph.add_node("mark_needs_human", mark_needs_human)

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
            "evaluate": "evaluate",
            "diagnose": "diagnose",
        },
    )
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "mark_success": "mark_success",
            "diagnose": "diagnose",
        },
    )
    graph.add_edge("mark_success", END)
    graph.add_conditional_edges(
        "diagnose",
        route_after_diagnose,
        {
            "plan": "plan",
            "feedback": "feedback",
        },
    )
    graph.add_conditional_edges(
        "feedback",
        route_after_feedback,
        {
            "plan": "plan",
            "mark_needs_human": "mark_needs_human",
        },
    )
    graph.add_edge("mark_needs_human", END)
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
    layer1 = test_result.get("deterministic_pass") is True
    layer2 = _layer2_passed(state.get("patch_evaluation") or {})
    passed = layer1 and layer2

    if status == "success" or passed:
        termination = "completed"
    elif status == "needs_human":
        termination = "needs_human"
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
        "structured_diagnosis": state.get("structured_diagnosis") or {},
        "attempt_history": list(state.get("attempt_history") or []),
        "patch_evaluation": state.get("patch_evaluation") or {},
        "test_result": test_result,
        "status": status,
        "llm_calls": telemetry.get("llm_calls", 0),
        "workflow_passed": passed,
        "relevant_files": state.get("relevant_files") or [],
        "retrieval_calls": telemetry.get("retrieval_calls", 0),
        "retry_count": int(state.get("retry_count") or 0),
        "human_retry_count": int(state.get("human_retry_count") or 0),
        "human_feedback": state.get("human_feedback") or "",
        "query_mode": telemetry.get("query_mode"),
        "embedder_name": telemetry.get("embedder_name"),
        "retrieve_query": telemetry.get("retrieve_query"),
    }


def run_workflow(
    *,
    client,
    issue: str,
    repo_path: str,
    test_command: str,
    lint_command: str = "ruff check app",
    model: str = "deepseek-v4-flash",
    max_steps: int = MAX_AGENT_STEPS,
    graph=None,
    sandbox=None,
    enable_search_code: bool = False,
    embedder_name: str = "hashing",
    query_mode: str = "issue",
    progress: ProgressReporter | None = None,
    feedback_provider=None,
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
                "lint_command": lint_command,
                "max_steps": max_steps,
                "sandbox": sandbox,
                "enable_search_code": enable_search_code,
                "embedder_name": embedder_name,
                "query_mode": query_mode,
                "progress": progress,
                "feedback_provider": feedback_provider,
            }
        },
    )
    result = adapt_result(final_state)
    # Wall-clock for the full V1 workflow (analyze/plan/execute/verify/evaluate/diagnose).
    result["latency"] = time.perf_counter() - started_at
    return result
