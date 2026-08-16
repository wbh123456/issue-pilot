"""Structured diagnose node and retry replanning."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes.diagnose import diagnose_failure
from agent.nodes.plan import structured_plan
from agent.state import DiagnosisValidationError, initial_state

VALID_PLAN = {
    "problem": "Expired JWT returns 500",
    "hypothesis": "decode_token misses ExpiredSignatureError",
    "files_to_inspect": ["app/auth.py"],
    "steps": ["inspect auth.py", "fix exception handling", "run tests"],
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
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()


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


VALID_DIAGNOSIS = {
    "root_cause": "decode_token misses ExpiredSignatureError",
    "failure_category": "WRONG_HYPOTHESIS",
    "new_hypothesis": "Expired JWT must map to 401",
    "next_actions": ["catch ExpiredSignatureError", "return 401"],
}

REVISED_PLAN = {
    "problem": "Expired JWT returns 500",
    "hypothesis": "Expired JWT must map to 401",
    "files_to_inspect": ["app/auth.py"],
    "steps": ["inspect auth.py", "catch expired JWT", "run tests"],
}


def _failed_state() -> dict:
    state = initial_state("Expired JWT returns 500")
    state["plan"] = VALID_PLAN
    state["test_result"] = {
        "command": "pytest -q",
        "lint_command": "ruff check app",
        "exit_code": 1,
        "deterministic_pass": False,
        "pytest_passed": False,
        "ruff_passed": True,
        "patch_valid": True,
        "output": "FAILED",
        "ruff_output": "ok",
        "git_diff": "--- status ---\n M app/auth.py",
        "changed_files": ["app/auth.py"],
        "untracked_files": [],
    }
    return state


class TestDiagnoseContract:
    def test_valid_json_updates_state_and_history(self) -> None:
        client = FakeClient([_Response(json.dumps(VALID_DIAGNOSIS))])
        out = diagnose_failure(_failed_state(), _config(client))
        assert out["structured_diagnosis"]["new_hypothesis"] == (
            VALID_DIAGNOSIS["new_hypothesis"]
        )
        assert VALID_DIAGNOSIS["root_cause"] in out["diagnosis"]
        assert out["retry_count"] == 1
        assert out["status"] == "retrying"
        assert len(out["attempt_history"]) == 1
        summary = out["attempt_history"][0]
        assert summary["attempt_index"] == 0
        assert summary["failure_source"] == "deterministic"
        assert summary["failure_category"] == "WRONG_HYPOTHESIS"
        assert summary["hypothesis"] == VALID_PLAN["hypothesis"]
        user = client.calls[0]["messages"][1]["content"]
        assert "Layer 1 result" in user
        assert "pytest_passed" in user
        assert "changed_files" in user
        assert "Changed files:" in user
        assert "- app/auth.py" in user
        assert "Git diff" in user
        assert user.index("Changed files:") < user.index("Git diff")
        assert "Prior attempts" in user

    def test_includes_prior_history_and_layer2(self) -> None:
        client = FakeClient([_Response(json.dumps(VALID_DIAGNOSIS))])
        state = _failed_state()
        state["retry_count"] = 1
        state["attempt_history"] = [
            {
                "attempt_index": 0,
                "hypothesis": "old idea",
                "deterministic_pass": False,
                "evaluator_pass": None,
                "failure_source": "deterministic",
                "failure_category": "BAD_PATCH",
                "root_cause": "first miss",
            }
        ]
        state["patch_evaluation"] = {
            "issue_resolved": False,
            "patch_scope": "too_narrow",
            "regression_risk": "low",
            "missing_tests": False,
            "feedback": "missed expired JWT",
        }
        state["test_result"]["deterministic_pass"] = True
        out = diagnose_failure(state, _config(client))
        user = client.calls[0]["messages"][1]["content"]
        assert "first miss" in user
        assert "too_narrow" in user
        assert out["attempt_history"][-1]["failure_source"] == "evaluator"
        assert out["retry_count"] == 2
        assert out["status"] == "failed"

    def test_layer1_fail_omits_stale_layer2_from_prompt(self) -> None:
        client = FakeClient([_Response(json.dumps(VALID_DIAGNOSIS))])
        state = _failed_state()
        state["patch_evaluation"] = {
            "issue_resolved": False,
            "patch_scope": "too_narrow",
            "regression_risk": "low",
            "missing_tests": False,
            "feedback": "missed expired JWT",
        }
        out = diagnose_failure(state, _config(client))
        user = client.calls[0]["messages"][1]["content"]
        assert "too_narrow" not in user
        assert "missed expired JWT" not in user
        assert out["attempt_history"][0]["failure_source"] == "deterministic"
        assert out["attempt_history"][0]["evaluator_pass"] is None
        assert state["patch_evaluation"]["patch_scope"] == "too_narrow"

    def test_reject_prose(self) -> None:
        client = FakeClient([_Response("the tests failed because of auth")])
        with pytest.raises(DiagnosisValidationError):
            diagnose_failure(_failed_state(), _config(client))


class TestReplan:
    def test_retry_plan_uses_diagnosis_and_changes_hypothesis(self) -> None:
        client = FakeClient([_Response(json.dumps(REVISED_PLAN))])
        state = initial_state("bug")
        state["analysis"] = "expired JWT"
        state["plan"] = VALID_PLAN
        state["retry_count"] = 1
        state["structured_diagnosis"] = VALID_DIAGNOSIS
        state["attempt_history"] = [
            {
                "attempt_index": 0,
                "hypothesis": VALID_PLAN["hypothesis"],
                "deterministic_pass": False,
                "evaluator_pass": None,
                "failure_source": "deterministic",
                "failure_category": "WRONG_HYPOTHESIS",
                "root_cause": VALID_DIAGNOSIS["root_cause"],
            }
        ]
        with (
            patch("agent.nodes.plan.list_files", return_value="app/auth.py"),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
        ):
            out = structured_plan(state, _config(client))

        assert out["plan"]["hypothesis"] == REVISED_PLAN["hypothesis"]
        assert out["plan"]["hypothesis"] != VALID_PLAN["hypothesis"]
        user = client.calls[0]["messages"][1]["content"]
        assert "This is a retry" in user
        assert VALID_DIAGNOSIS["new_hypothesis"] in user
        assert VALID_PLAN["hypothesis"] in user

    def test_retry_overrides_repeated_hypothesis(self) -> None:
        client = FakeClient([_Response(json.dumps(VALID_PLAN))])
        state = initial_state("bug")
        state["plan"] = VALID_PLAN
        state["retry_count"] = 1
        state["structured_diagnosis"] = VALID_DIAGNOSIS
        with (
            patch("agent.nodes.plan.list_files", return_value="app/auth.py"),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
        ):
            out = structured_plan(state, _config(client))
        assert out["plan"]["hypothesis"] == VALID_DIAGNOSIS["new_hypothesis"]
        assert out["plan"]["hypothesis"] != VALID_PLAN["hypothesis"]
