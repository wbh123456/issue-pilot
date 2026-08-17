"""Diagnose node: structured failure analysis before a possible replan."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.nodes.plan import _strip_code_fence
from agent.state import (
    AgentState,
    AttemptSummary,
    DiagnosisValidationError,
    EvaluationValidationError,
    FailureSource,
    parse_attempt_history,
    parse_structured_diagnosis,
    patch_evaluation_passed,
)
from agent.tools.git import format_file_lists
from harness.limits import AGENT_TEMPERATURE, MAX_RETRY, truncate_output

from ._runtime import get_reporter, merge_telemetry, require_config, stage_usage

_DIAGNOSE_CONTEXT_LIMIT = 4000

DIAGNOSE_SYSTEM = """
You are diagnosing why a coding-agent attempt failed verification.

Return ONLY a single JSON object (no markdown, no prose) with exactly these keys:
{
  "root_cause": string,
  "failure_category": "RETRIEVAL_FAILURE" | "WRONG_HYPOTHESIS" | "BAD_PATCH" | "TEST_FAILURE" | "ENVIRONMENT_FAILURE",
  "new_hypothesis": string,
  "next_actions": string[]
}

Rules:
- root_cause and new_hypothesis must be non-empty
- new_hypothesis must differ from the failed plan hypothesis
- next_actions must be 1 to 5 concrete follow-ups for the planner/executor
- do not call tools
- do not edit code
""".strip()


def _clip(text: object) -> str:
    return truncate_output(str(text or ""), _DIAGNOSE_CONTEXT_LIMIT)


def _evaluator_pass(raw: object) -> bool | None:
    if not raw:
        return None
    try:
        return patch_evaluation_passed(raw)
    except EvaluationValidationError:
        return False


def _failure_source(
    test_result: dict, evaluator_pass: bool | None
) -> FailureSource:
    if test_result.get("deterministic_pass") is True and evaluator_pass is False:
        return "evaluator"
    return "deterministic"


def _diagnosis_text(parsed) -> str:
    actions = "; ".join(parsed.next_actions)
    return (
        f"{parsed.root_cause} [{parsed.failure_category}] "
        f"New hypothesis: {parsed.new_hypothesis} Next: {actions}"
    )


def _layer1_payload(test_result: dict) -> dict:
    return {
        "command": test_result.get("command"),
        "lint_command": test_result.get("lint_command"),
        "exit_code": test_result.get("exit_code"),
        "deterministic_pass": test_result.get("deterministic_pass"),
        "pytest_passed": test_result.get("pytest_passed"),
        "ruff_passed": test_result.get("ruff_passed"),
        "ruff_autofixed": test_result.get("ruff_autofixed"),
        "patch_valid": test_result.get("patch_valid"),
        "changed_files": list(test_result.get("changed_files") or []),
        "untracked_files": list(test_result.get("untracked_files") or []),
        "output": _clip(test_result.get("output")),
        "ruff_output": _clip(test_result.get("ruff_output")),
    }


def diagnose_failure(state: AgentState, config: RunnableConfig) -> dict:
    cfg = require_config(config, "client", "model")
    client = cfg["client"]
    model = cfg["model"]
    test_result = dict(state.get("test_result") or {})
    plan = state.get("plan") or {}
    history = parse_attempt_history(list[dict[str, Any]](state.get("attempt_history") or []))
    layer1_pass = test_result.get("deterministic_pass") is True
    layer2 = (state.get("patch_evaluation") or {}) if layer1_pass else {}
    evaluator_pass = _evaluator_pass(layer2)
    failure_source = _failure_source(test_result, evaluator_pass)

    reviewer = (state.get("human_feedback") or "").strip()
    reviewer_block = (
        f"Reviewer feedback:\n{_clip(reviewer)}\n\n" if reviewer else ""
    )
    user_content = (
        f"Issue:\n{state['issue']}\n\n"
        f"{reviewer_block}"
        f"Current plan:\n{_clip(json.dumps(plan, ensure_ascii=False, indent=2))}\n\n"
        f"Layer 1 result:\n{_clip(json.dumps(_layer1_payload(test_result), ensure_ascii=False, indent=2))}\n\n"
        f"{format_file_lists(test_result.get('changed_files'), test_result.get('untracked_files'))}\n\n"
        f"Git diff:\n{_clip(test_result.get('git_diff') or '(none)')}\n\n"
        f"Layer 2 evaluation:\n{_clip(json.dumps(layer2, ensure_ascii=False))}\n\n"
        f"Prior attempts:\n{_clip(json.dumps([item.model_dump() for item in history], ensure_ascii=False, indent=2))}\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DIAGNOSE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=AGENT_TEMPERATURE,
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise DiagnosisValidationError("Diagnosis is empty")
    parsed = parse_structured_diagnosis(_strip_code_fence(raw))

    attempt_index = int(state.get("retry_count") or 0)
    summary = AttemptSummary(
        attempt_index=attempt_index,
        hypothesis=str((plan or {}).get("hypothesis") or ""),
        deterministic_pass=bool(test_result.get("deterministic_pass")),
        evaluator_pass=evaluator_pass,
        failure_source=failure_source,
        failure_category=parsed.failure_category,
        root_cause=parsed.root_cause,
    )
    history = parse_attempt_history([*history, summary])
    retry_count = attempt_index + 1
    status = "failed" if retry_count >= MAX_RETRY else "retrying"
    diagnosis = _diagnosis_text(parsed)
    get_reporter(config).stage("diagnose", diagnosis)
    return {
        "diagnosis": diagnosis,
        "structured_diagnosis": parsed.model_dump(),
        "attempt_history": [item.model_dump() for item in history],
        "retry_count": retry_count,
        "status": status,
        "telemetry": merge_telemetry(state, **stage_usage("diagnose", response)),
    }
