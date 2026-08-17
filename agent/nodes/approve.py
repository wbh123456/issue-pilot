"""Approval gate: optional LangGraph interrupt after Layer 1 and Layer 2 pass."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from agent.state import AgentState
from harness.limits import truncate_output

from ._runtime import configurable, get_reporter

ApprovalDecision = Literal["approve", "reject", "feedback"]

_DECISIONS = {"approve", "reject", "feedback"}
_TEST_RESULT_KEYS = (
    "deterministic_pass",
    "pytest_passed",
    "ruff_passed",
    "patch_valid",
    "exit_code",
    "output",
)


class ApprovalError(RuntimeError):
    """Raised when approval is requested without a durable checkpointer."""


def review_payload(state: AgentState) -> dict[str, Any]:
    """Six-part review bundle for the CLI (issue, plan, files, diff, tests, eval)."""
    test_result = dict(state.get("test_result") or {})
    compact_tests = {
        key: test_result.get(key) for key in _TEST_RESULT_KEYS if key in test_result
    }
    if "output" in compact_tests:
        compact_tests["output"] = truncate_output(str(compact_tests.get("output") or ""))
    return {
        "issue": state.get("issue") or "",
        "plan": dict(state.get("plan") or {}),
        "changed_files": list(test_result.get("changed_files") or []),
        "git_diff": truncate_output(str(test_result.get("git_diff") or "")),
        "test_result": compact_tests,
        "evaluator_result": dict(state.get("patch_evaluation") or {}),
    }


def interrupt_payload(result: Any) -> dict[str, Any] | None:
    """Extract the review payload from an interrupted invoke result, if any."""
    if not isinstance(result, dict):
        return None
    items = result.get("__interrupt__")
    if not items:
        return None
    first = items[0]
    value = getattr(first, "value", first)
    if not isinstance(value, dict):
        return None
    return dict(value)


def parse_approval_resume(raw: Any) -> tuple[ApprovalDecision, str]:
    """Normalize ``Command(resume=...)`` to ``(decision, feedback)``.

    Unknown or blank feedback fail closed to ``reject``.
    """
    decision = ""
    feedback = ""
    if isinstance(raw, str):
        decision = raw.strip().lower()
    elif isinstance(raw, dict):
        decision = str(raw.get("decision") or "").strip().lower()
        feedback = str(raw.get("feedback") or "").strip()
    if decision not in _DECISIONS:
        return "reject", ""
    if decision == "feedback" and not feedback:
        return "reject", ""
    if decision != "feedback":
        return decision, ""  # type: ignore[return-value]
    return "feedback", feedback


def _has_checkpointer(config: RunnableConfig) -> bool:
    return configurable(config).get("__pregel_checkpointer") is not None


def await_approval(state: AgentState, config: RunnableConfig) -> dict:
    """Pass through unless ``require_approval``; otherwise interrupt for a decision."""
    if not configurable(config).get("require_approval"):
        return {}
    if not _has_checkpointer(config):
        raise ApprovalError(
            "require_approval needs a checkpointer; interrupt() cannot resume without one"
        )

    payload = review_payload(state)
    get_reporter(config).stage("approval", "waiting")
    decision, feedback = parse_approval_resume(interrupt(payload))
    history = list(state.get("approval_history") or [])
    history.append({"decision": decision, "feedback": feedback})
    update: dict[str, Any] = {
        "approval_decision": decision,
        "approval_history": history,
    }
    if decision == "approve":
        update["status"] = "approved"
        get_reporter(config).stage("approval", "approved")
    elif decision == "reject":
        update["status"] = "rejected"
        get_reporter(config).stage("approval", "rejected")
    else:
        update["status"] = "approval_feedback"
        update["human_feedback"] = feedback
        get_reporter(config).stage("approval", "feedback")
    return update
