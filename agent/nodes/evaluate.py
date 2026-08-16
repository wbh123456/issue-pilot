"""Evaluate node: Layer 2 LLM-as-judge after Layer 1 passes."""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig

from agent.nodes.plan import _strip_code_fence
from agent.state import (
    AgentState,
    EvaluationValidationError,
    parse_patch_evaluation,
    patch_evaluation_passed,
)
from agent.tools.git import format_file_lists
from harness.limits import AGENT_TEMPERATURE, truncate_output

from ._runtime import get_reporter, merge_telemetry, require_config, stage_usage

_EVALUATE_CONTEXT_LIMIT = 4000

EVALUATE_SYSTEM = """
You are judging whether a coding-agent patch resolved the reported issue.

Return ONLY a single JSON object (no markdown, no prose) with exactly these keys:
{
  "issue_resolved": bool,
  "patch_scope": "appropriate" | "too_broad" | "too_narrow" | "unrelated",
  "regression_risk": "low" | "medium" | "high",
  "missing_tests": bool,
  "feedback": string
}

Rules:
- do not include a pass, passed, or score field; pass/fail is computed elsewhere
- issue_resolved is whether the issue itself is fixed, not whether tests ran
- patch_scope describes how the diff relates to the issue
- regression_risk is low only when the change is tightly scoped and unlikely to break other behavior
- missing_tests is true when the patch needed new or updated tests and did not add them
- feedback may be empty; keep it under 800 characters
- do not call tools
- do not edit code
""".strip()


def _clip(text: object) -> str:
    return truncate_output(str(text or ""), _EVALUATE_CONTEXT_LIMIT)


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


def evaluate_patch(state: AgentState, config: RunnableConfig) -> dict:
    test_result = dict(state.get("test_result") or {})
    if test_result.get("deterministic_pass") is not True:
        get_reporter(config).stage("evaluate", "skipped")
        return {
            "patch_evaluation": {},
            "status": "evaluate_skipped",
        }

    cfg = require_config(config, "client", "model")
    client = cfg["client"]
    model = cfg["model"]
    plan = state.get("plan") or {}
    user_content = (
        f"Issue:\n{state['issue']}\n\n"
        f"Current plan:\n{_clip(json.dumps(plan, ensure_ascii=False, indent=2))}\n\n"
        f"Layer 1 result:\n{_clip(json.dumps(_layer1_payload(test_result), ensure_ascii=False, indent=2))}\n\n"
        f"{format_file_lists(test_result.get('changed_files'), test_result.get('untracked_files'))}\n\n"
        f"Git diff:\n{_clip(test_result.get('git_diff') or '(none)')}\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EVALUATE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=AGENT_TEMPERATURE,
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise EvaluationValidationError("Evaluation is empty")
    parsed = parse_patch_evaluation(_strip_code_fence(raw))
    passed = patch_evaluation_passed(parsed)
    label = "PASS" if passed else "FAIL"
    get_reporter(config).stage(
        "evaluate",
        f"{label}  scope={parsed.patch_scope} risk={parsed.regression_risk}",
    )
    return {
        "patch_evaluation": parsed.model_dump(),
        "status": "evaluate_passed" if passed else "evaluate_failed",
        "telemetry": merge_telemetry(state, **stage_usage("evaluate", response)),
    }
