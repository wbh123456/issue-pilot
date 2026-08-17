"""Layer 2 LLM evaluator: schema-constrained judgment, mechanical pass."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.graph import adapt_result, route_after_evaluate, route_after_verify
from agent.nodes.evaluate import evaluate_patch
from agent.state import EvaluationValidationError, initial_state, patch_evaluation_passed


VALID_PLAN = {
    "problem": "Expired JWT returns 500",
    "hypothesis": "decode_token misses ExpiredSignatureError",
    "files_to_inspect": ["app/auth.py"],
    "steps": ["inspect auth.py", "fix exception handling", "run tests"],
}

VALID_EVALUATION = {
    "issue_resolved": True,
    "patch_scope": "appropriate",
    "regression_risk": "low",
    "missing_tests": False,
    "feedback": "",
}


def _config(client: FakeClient, **extra: Any) -> dict:
    configurable = {
        "client": client,
        "model": "fake-model",
        "repo_path": "/tmp/repo",
        "test_command": "pytest -q",
        "max_steps": 5,
    }
    configurable.update(extra)
    return {"configurable": configurable}


class _Usage:
    def __init__(self, prompt: int = 10, completion: int = 5) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str, *, prompt: int = 10, completion: int = 5) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage(prompt, completion)


class FakeClient:
    def __init__(self, responses: list[_Response]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = MagicMock()
        self.chat.completions.create.side_effect = self._create

    def _create(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError(f"unexpected LLM call: {kwargs}")
        return self._responses.pop(0)


def _layer1_pass_state() -> dict:
    state = initial_state("Expired JWT returns 500")
    state["plan"] = VALID_PLAN
    state["test_result"] = {
        "command": "pytest -q",
        "lint_command": "ruff check app",
        "exit_code": 0,
        "deterministic_pass": True,
        "pytest_passed": True,
        "ruff_passed": True,
        "patch_valid": True,
        "output": "ok",
        "ruff_output": "ok",
        "git_diff": "--- status ---\n M app/auth.py\n--- diff ---\n+fixed\n",
        "changed_files": ["app/auth.py"],
        "untracked_files": [],
    }
    return state


class TestEvaluateNode:
    def test_positive_judgment_passes_and_attributes_tokens(self) -> None:
        client = FakeClient([_Response(json.dumps(VALID_EVALUATION))])
        out = evaluate_patch(_layer1_pass_state(), _config(client))
        assert out["status"] == "evaluate_passed"
        assert out["patch_evaluation"] == VALID_EVALUATION
        assert patch_evaluation_passed(out["patch_evaluation"]) is True
        assert "passed" not in out["patch_evaluation"]
        assert out["telemetry"]["stage_tokens"]["evaluate"]["llm_calls"] == 1
        assert out["telemetry"]["stage_tokens"]["evaluate"]["prompt_tokens"] == 10
        assert out["telemetry"]["llm_calls"] == 1
        user = client.calls[0]["messages"][1]["content"]
        assert "Expired JWT" in user
        assert "Layer 1 result" in user
        assert "Changed files:" in user
        assert "- app/auth.py" in user
        assert "Git diff" in user
        assert user.index("Changed files:") < user.index("Git diff")
        assert user.index("- app/auth.py") < user.index("Git diff")
        system = client.calls[0]["messages"][0]["content"]
        assert "pass" in system.lower()
        assert "do not include a pass" in system.lower()

    def test_accepts_fenced_json(self) -> None:
        fenced = "```json\n" + json.dumps(VALID_EVALUATION) + "\n```"
        client = FakeClient([_Response(fenced)])
        out = evaluate_patch(_layer1_pass_state(), _config(client))
        assert out["status"] == "evaluate_passed"
        assert out["patch_evaluation"]["issue_resolved"] is True

    def test_reject_prose_and_pass_field(self) -> None:
        client = FakeClient([_Response("the patch looks good")])
        with pytest.raises(EvaluationValidationError):
            evaluate_patch(_layer1_pass_state(), _config(client))
        client = FakeClient(
            [_Response(json.dumps({**VALID_EVALUATION, "passed": True}))]
        )
        with pytest.raises(EvaluationValidationError):
            evaluate_patch(_layer1_pass_state(), _config(client))
        client = FakeClient([_Response("")])
        with pytest.raises(EvaluationValidationError):
            evaluate_patch(_layer1_pass_state(), _config(client))

    @pytest.mark.parametrize(
        "override",
        [
            {"issue_resolved": False},
            {"patch_scope": "too_broad"},
            {"patch_scope": "too_narrow"},
            {"patch_scope": "unrelated"},
            {"regression_risk": "medium"},
            {"regression_risk": "high"},
            {"missing_tests": True},
        ],
    )
    def test_each_blocking_negative_fails_mechanically(self, override: dict) -> None:
        payload = {**VALID_EVALUATION, **override}
        client = FakeClient([_Response(json.dumps(payload))])
        out = evaluate_patch(_layer1_pass_state(), _config(client))
        assert out["status"] == "evaluate_failed"
        assert patch_evaluation_passed(out["patch_evaluation"]) is False
        assert out["telemetry"]["stage_tokens"]["evaluate"]["llm_calls"] == 1

    def test_layer1_failure_skips_llm_and_cannot_pass(self) -> None:
        client = FakeClient([_Response(json.dumps(VALID_EVALUATION))])
        state = _layer1_pass_state()
        state["test_result"]["deterministic_pass"] = False
        out = evaluate_patch(state, _config(client))
        assert client.calls == []
        assert out["status"] == "evaluate_skipped"
        assert out["patch_evaluation"] == {}
        assert "telemetry" not in out


class TestEvaluateRouting:
    def test_verify_pass_goes_to_evaluate(self) -> None:
        assert (
            route_after_verify({"test_result": {"deterministic_pass": True}})
            == "evaluate"
        )
        assert (
            route_after_verify({"test_result": {"deterministic_pass": False}})
            == "diagnose"
        )
        assert route_after_verify({}) == "diagnose"

    def test_evaluate_pass_goes_to_approval_gate(self) -> None:
        assert (
            route_after_evaluate(
                {
                    "test_result": {"deterministic_pass": True},
                    "patch_evaluation": VALID_EVALUATION,
                }
            )
            == "await_approval"
        )

    def test_evaluate_reject_goes_to_diagnose(self) -> None:
        reject = {**VALID_EVALUATION, "issue_resolved": False}
        assert (
            route_after_evaluate(
                {
                    "test_result": {"deterministic_pass": True},
                    "patch_evaluation": reject,
                }
            )
            == "diagnose"
        )
        assert (
            route_after_evaluate({"test_result": {"deterministic_pass": True}})
            == "diagnose"
        )

    def test_layer1_fail_never_succeeds_even_with_green_eval(self) -> None:
        assert (
            route_after_evaluate(
                {
                    "test_result": {"deterministic_pass": False},
                    "patch_evaluation": VALID_EVALUATION,
                }
            )
            == "diagnose"
        )


class TestAdaptResultLayer2:
    def test_workflow_passed_requires_both_layers(self) -> None:
        state = initial_state("bug")
        state["status"] = "evaluate_passed"
        state["test_result"] = {"deterministic_pass": True}
        state["patch_evaluation"] = VALID_EVALUATION
        assert adapt_result(state)["workflow_passed"] is True

        state["patch_evaluation"] = {**VALID_EVALUATION, "missing_tests": True}
        assert adapt_result(state)["workflow_passed"] is False

        state["patch_evaluation"] = VALID_EVALUATION
        state["test_result"] = {"deterministic_pass": False}
        assert adapt_result(state)["workflow_passed"] is False

        state["test_result"] = {"deterministic_pass": True}
        state["patch_evaluation"] = {}
        assert adapt_result(state)["workflow_passed"] is False
