"""Serializable V1 workflow state and structured planning contract."""

from __future__ import annotations

import json
from typing import Any, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PlanValidationError(ValueError):
    """Raised when planner output does not match the structured plan schema."""


class StructuredPlan(BaseModel):
    """Planner output contract used by the V1 plan node."""

    model_config = ConfigDict(extra="forbid")

    problem: str = Field(min_length=1, max_length=500)
    hypothesis: str = Field(min_length=1, max_length=800)
    files_to_inspect: list[str] = Field(default_factory=list, max_length=8)
    steps: list[str] = Field(min_length=3, max_length=5)


class Telemetry(TypedDict, total=False):
    tool_call_count: int
    file_reads: int
    prompt_tokens: int
    completion_tokens: int
    tokens: int
    llm_calls: int
    steps: int
    latency: float
    # Executor artifacts folded into telemetry for the minimal AgentState.
    final_answer: str
    termination: str
    trajectory: list[Any]
    messages: list[Any]
    # Per-stage usage: analyze / plan / execute / diagnose.
    stage_tokens: dict[str, dict[str, int]]


# Optional: node visit log for debugging/eval; not required for routing.
# class WorkflowTraceEvent(TypedDict, total=False):
#     node: str
#     status: str
#     detail: str


class AgentState(TypedDict):
    """JSON-serializable LangGraph state for the V1 Plan-Execute workflow.

    Runtime-only objects (LLM client, repo path, test command) stay outside
    this state so it remains checkpoint-ready.
    """

    # --- must-keep (Phase 2 DoD) ---
    issue: str
    analysis: NotRequired[str]
    plan: NotRequired[dict[str, Any]]
    test_result: NotRequired[dict[str, Any]]
    diagnosis: NotRequired[str]
    status: NotRequired[str]
    telemetry: NotRequired[Telemetry]

    # --- optional (commented out to keep Phase 2 state minimal) ---
    # Can be derived from plan.files_to_inspect:
    # relevant_files: NotRequired[list[str]]
    # Can be derived from verify git_diff:
    # changed_files: NotRequired[list[str]]
    # Executor metrics can fold into telemetry + run JSON:
    # execution_result: NotRequired[dict[str, Any]]
    # Phase 2 freezes retries at 0; enable for Day 5:
    # retry_count: NotRequired[int]
    # Debug/eval aid; not required for routing correctness:
    # workflow_trace: NotRequired[list[WorkflowTraceEvent]]


def parse_structured_plan(data: Any) -> StructuredPlan:
    """Parse and validate a structured plan; reject malformed input explicitly."""
    if isinstance(data, StructuredPlan):
        return data

    if isinstance(data, str):
        text = data.strip()
        if not text:
            raise PlanValidationError("Plan is empty")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanValidationError(
                f"Plan is not valid JSON: {exc}"
            ) from exc

    if not isinstance(data, dict):
        raise PlanValidationError(
            f"Plan must be a JSON object, got {type(data).__name__}"
        )

    try:
        return StructuredPlan.model_validate(data)
    except ValidationError as exc:
        raise PlanValidationError(f"Malformed plan: {exc}") from exc


def initial_state(issue: str) -> AgentState:
    """Return a fresh AgentState for a single issue run."""
    return {
        "issue": issue,
        "analysis": "",
        "plan": {},
        "test_result": {},
        "diagnosis": "",
        "status": "pending",
        "telemetry": {
            "tool_call_count": 0,
            "file_reads": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tokens": 0,
            "llm_calls": 0,
            "steps": 0,
            "latency": 0.0,
            "stage_tokens": {},
        },
        # Optional fields (kept commented with AgentState):
        # "relevant_files": [],
        # "changed_files": [],
        # "execution_result": {},
        # "retry_count": 0,
        # "workflow_trace": [],
    }
