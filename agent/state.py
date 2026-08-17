"""Serializable V1 workflow state and structured planning / recovery contracts."""

from __future__ import annotations

import json
from typing import Any, Literal, NotRequired, TypeVar, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TModel = TypeVar("TModel", bound=BaseModel)

FailureCategory = Literal[
    "RETRIEVAL_FAILURE",
    "WRONG_HYPOTHESIS",
    "BAD_PATCH",
    "TEST_FAILURE",
    "ENVIRONMENT_FAILURE",
]
FailureSource = Literal["deterministic", "evaluator"]
PatchScope = Literal["appropriate", "too_broad", "too_narrow", "unrelated"]
RegressionRisk = Literal["low", "medium", "high"]


class PlanValidationError(ValueError):
    """Raised when planner output does not match the structured plan schema."""


class DiagnosisValidationError(ValueError):
    """Raised when diagnosis output does not match the structured schema."""


class AttemptSummaryValidationError(ValueError):
    """Raised when an attempt summary or history list is malformed."""


class EvaluationValidationError(ValueError):
    """Raised when Layer 2 evaluator output does not match the schema."""


class StructuredPlan(BaseModel):
    """Planner output contract used by the V1 plan node."""

    model_config = ConfigDict(extra="forbid")

    problem: str = Field(min_length=1, max_length=500)
    hypothesis: str = Field(min_length=1, max_length=800)
    files_to_inspect: list[str] = Field(default_factory=list, max_length=8)
    steps: list[str] = Field(min_length=3, max_length=5)


class StructuredDiagnosis(BaseModel):
    """Validated diagnose-node output. Nodes populate this in a later step."""

    model_config = ConfigDict(extra="forbid")

    root_cause: str = Field(min_length=1, max_length=800)
    failure_category: FailureCategory
    new_hypothesis: str = Field(min_length=1, max_length=800)
    next_actions: list[str] = Field(min_length=1, max_length=5)


class PatchEvaluation(BaseModel):
    """Layer 2 LLM-as-judge fields. Pass/fail is computed mechanically."""

    model_config = ConfigDict(extra="forbid")

    issue_resolved: bool
    patch_scope: PatchScope
    regression_risk: RegressionRisk
    missing_tests: bool
    feedback: str = Field(default="", max_length=800)


class AttemptSummary(BaseModel):
    """Bounded per-attempt record for diagnose context and reporting."""

    model_config = ConfigDict(extra="forbid")

    attempt_index: int = Field(ge=0, le=2)
    hypothesis: str = Field(default="", max_length=800)
    deterministic_pass: bool
    evaluator_pass: bool | None = None
    failure_source: FailureSource | None = None
    failure_category: FailureCategory | None = None
    root_cause: str = Field(default="", max_length=800)


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
    # Per-stage usage: analyze / retrieve / plan / execute / evaluate / diagnose.
    stage_tokens: dict[str, dict[str, int]]
    retrieval_calls: int


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
    relevant_files: NotRequired[list[str]]
    retrieved_context: NotRequired[str]
    retry_count: NotRequired[int]
    structured_diagnosis: NotRequired[dict[str, Any]]
    attempt_history: NotRequired[list[dict[str, Any]]]
    patch_evaluation: NotRequired[dict[str, Any]]
    human_retry_count: NotRequired[int]
    human_feedback: NotRequired[str]
    approval_decision: NotRequired[str]
    approval_history: NotRequired[list[dict[str, Any]]]

    # --- optional (commented out to keep Phase 2 state minimal) ---
    # Can be derived from verify git_diff:
    # changed_files: NotRequired[list[str]]
    # Executor metrics can fold into telemetry + run JSON:
    # execution_result: NotRequired[dict[str, Any]]
    # Debug/eval aid; not required for routing correctness:
    # workflow_trace: NotRequired[list[WorkflowTraceEvent]]


def _load_json_value(
    data: Any,
    *,
    empty_message: str,
    error_cls: type[ValueError],
) -> Any:
    """Decode a JSON string; pass through already-parsed values."""
    if not isinstance(data, str):
        return data
    text = data.strip()
    if not text:
        raise error_cls(empty_message)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise error_cls(f"Not valid JSON: {exc}") from exc


def _parse_model(
    model_cls: type[TModel],
    data: Any,
    *,
    error_cls: type[ValueError],
    empty_message: str,
    type_message: str,
    malformed_prefix: str,
) -> TModel:
    data = _load_json_value(data, empty_message=empty_message, error_cls=error_cls)
    if not isinstance(data, dict):
        raise error_cls(f"{type_message}, got {type(data).__name__}")
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise error_cls(f"{malformed_prefix}: {exc}") from exc


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


def parse_structured_diagnosis(data: Any) -> StructuredDiagnosis:
    """Parse and validate a structured diagnosis; reject malformed input."""
    if isinstance(data, StructuredDiagnosis):
        return data
    return _parse_model(
        StructuredDiagnosis,
        data,
        error_cls=DiagnosisValidationError,
        empty_message="Diagnosis is empty",
        type_message="Diagnosis must be a JSON object",
        malformed_prefix="Malformed diagnosis",
    )


def parse_patch_evaluation(data: Any) -> PatchEvaluation:
    """Parse and validate Layer 2 evaluator output; reject malformed input."""
    if isinstance(data, PatchEvaluation):
        return data
    return _parse_model(
        PatchEvaluation,
        data,
        error_cls=EvaluationValidationError,
        empty_message="Evaluation is empty",
        type_message="Evaluation must be a JSON object",
        malformed_prefix="Malformed evaluation",
    )


def parse_attempt_summary(data: Any) -> AttemptSummary:
    """Parse and validate a single attempt summary."""
    if isinstance(data, AttemptSummary):
        return data
    return _parse_model(
        AttemptSummary,
        data,
        error_cls=AttemptSummaryValidationError,
        empty_message="Attempt summary is empty",
        type_message="Attempt summary must be a JSON object",
        malformed_prefix="Malformed attempt summary",
    )


def parse_attempt_history(data: Any) -> list[AttemptSummary]:
    """Parse a bounded list of attempt summaries (max 3)."""
    if isinstance(data, list) and all(
        isinstance(item, AttemptSummary) for item in data
    ):
        if len(data) > 3:
            raise AttemptSummaryValidationError(
                "Attempt history exceeds max_length=3"
            )
        return data

    loaded = _load_json_value(
        data,
        empty_message="Attempt history is empty",
        error_cls=AttemptSummaryValidationError,
    )
    if not isinstance(loaded, list):
        raise AttemptSummaryValidationError(
            f"Attempt history must be a JSON array, got {type(loaded).__name__}"
        )
    if len(loaded) > 3:
        raise AttemptSummaryValidationError(
            "Attempt history exceeds max_length=3"
        )
    return [parse_attempt_summary(item) for item in loaded]


def patch_evaluation_passed(data: Any) -> bool:
    """Mechanical Layer 2 pass. The model cannot emit or override this flag."""
    evaluation = parse_patch_evaluation(data)
    return (
        evaluation.issue_resolved
        and evaluation.patch_scope == "appropriate"
        and evaluation.regression_risk == "low"
        and evaluation.missing_tests is False
    )


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
            "retrieval_calls": 0,
            "stage_tokens": {},
        },
        "relevant_files": [],
        "retrieved_context": "",
        "retry_count": 0,
        "structured_diagnosis": {},
        "attempt_history": [],
        "patch_evaluation": {},
        "human_retry_count": 0,
        "human_feedback": "",
        "approval_decision": "",
        "approval_history": [],
    }
