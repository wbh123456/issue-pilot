"""Mocked tests for V1 LangGraph workflow, plan validation, and V0 seam."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.graph import (
    adapt_result,
    build_graph,
    route_after_approval,
    route_after_diagnose,
    route_after_evaluate,
    route_after_feedback,
    route_after_verify,
    run_workflow,
)
from agent.loop import run_agent
from agent.nodes.execute import _workflow_context
from agent.nodes.plan import _strip_code_fence, structured_plan
from agent.state import (
    PlanValidationError,
    initial_state,
    parse_structured_plan,
    patch_evaluation_passed,
)
from agent.tools.git import WorktreeDiff
from agent.tools.shell import CommandOutcome


class _Usage:
    def __init__(self, prompt: int = 10, completion: int = 5) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Message:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.message = _Message(content, tool_calls)


class _Response:
    def __init__(
        self,
        content: str,
        *,
        prompt: int = 10,
        completion: int = 5,
        tool_calls: list | None = None,
    ) -> None:
        self.choices = [_Choice(content, tool_calls)]
        self.usage = _Usage(prompt, completion)


class FakeClient:
    """OpenAI-compatible client that returns scripted chat completions."""

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


class RecordingReporter:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def stage(self, name: str, detail: str = "") -> None:
        self.events.append(("stage", name, detail))

    def note(self, text: str) -> None:
        self.events.append(("note", text))

    def tool(self, step: int, name: str, args: dict[str, Any], result: str) -> None:
        self.events.append(("tool", step, name, args, result))


VALID_PLAN = {
    "problem": "Expired JWT returns 500",
    "hypothesis": "decode_token misses ExpiredSignatureError",
    "files_to_inspect": ["app/auth.py"],
    "steps": ["inspect auth.py", "fix exception handling", "run tests"],
}

VALID_DIAGNOSIS = {
    "root_cause": "Root cause: wrong exception handler.",
    "failure_category": "WRONG_HYPOTHESIS",
    "new_hypothesis": "Need to catch ExpiredSignatureError",
    "next_actions": ["inspect auth.py", "catch expired JWT", "re-run tests"],
}

STILL_FAILING_DIAGNOSIS = {
    "root_cause": "Still failing: patch did not cover the gold case.",
    "failure_category": "BAD_PATCH",
    "new_hypothesis": "Also handle the missing user_id claim",
    "next_actions": ["read auth claims", "return 401", "re-run tests"],
}

RETRY_DIAGNOSIS = {
    "root_cause": "Retry with a 409 before indexing slots.",
    "failure_category": "WRONG_HYPOTHESIS",
    "new_hypothesis": "Return 409 when stock is insufficient",
    "next_actions": ["guard allocate_bin", "return 409", "keep stock"],
}

REVISED_PLAN = {
    **VALID_PLAN,
    "hypothesis": "Need to catch ExpiredSignatureError",
    "steps": ["inspect auth.py", "catch expired JWT", "re-run tests"],
}

RETRY_PLAN = {
    **VALID_PLAN,
    "hypothesis": "Return 409 when stock is insufficient",
    "steps": ["guard allocate_bin", "return 409", "keep stock"],
}

VALID_EVALUATION = {
    "issue_resolved": True,
    "patch_scope": "appropriate",
    "regression_risk": "low",
    "missing_tests": False,
    "feedback": "",
}

REJECT_EVALUATION = {
    "issue_resolved": False,
    "patch_scope": "too_narrow",
    "regression_risk": "low",
    "missing_tests": False,
    "feedback": "missed ExpiredSignatureError",
}

HUMAN_FEEDBACK = "Catch ExpiredSignatureError and return 401"

FEEDBACK_PLAN = {
    **VALID_PLAN,
    "hypothesis": "Catch ExpiredSignatureError and return 401",
    "steps": ["inspect auth.py", "catch expired JWT", "re-run tests"],
}

_FAKE_INVENTORY = {
    ".": "app/\ntests/\nREADME.md",
    "app": "app/auth.py\napp/main.py",
    "tests": "tests/test_auth.py",
}


def _fake_list_files(repo_path: str, path: str = ".") -> str:
    return _FAKE_INVENTORY.get(path, "Error: path not found")


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


PASS_PATCH = WorktreeDiff(
    status=" M app/auth.py",
    diff="diff --git a/app/auth.py b/app/auth.py\n+fixed\n",
    changed_files=["app/auth.py"],
)
EMPTY_PATCH = WorktreeDiff()


def _command_outcome(command: str, *, ok: bool) -> CommandOutcome:
    argv = command.split()
    return CommandOutcome(
        command=argv,
        exit_code=0 if ok else 1,
        stdout="ok" if ok else "FAILED",
    )


def _layer1_run_command(*, pytest_ok: bool = True, ruff_ok: bool = True):
    def _run(repo_path: str, command: str, **kwargs: Any) -> CommandOutcome:
        is_ruff = command.strip().startswith("ruff")
        return _command_outcome(command, ok=ruff_ok if is_ruff else pytest_ok)

    return _run


def _fail_n_then_pass_commands(failures: int):
    pytest_calls = {"n": 0}

    def _run(repo_path: str, command: str, **kwargs: Any) -> CommandOutcome:
        if command.strip().startswith("ruff"):
            return _command_outcome(command, ok=True)
        pytest_calls["n"] += 1
        return _command_outcome(command, ok=pytest_calls["n"] > failures)

    return _run


def _fail_then_pass_commands():
    return _fail_n_then_pass_commands(1)


class TestPlanValidation:
    def test_valid_plan_and_fence(self) -> None:
        plan = parse_structured_plan(VALID_PLAN)
        assert plan.files_to_inspect == ["app/auth.py"]
        fenced = "```json\n" + json.dumps(VALID_PLAN) + "\n```"
        assert parse_structured_plan(_strip_code_fence(fenced)).steps == VALID_PLAN["steps"]

    def test_reject_prose_missing_fields_and_extras(self) -> None:
        with pytest.raises(PlanValidationError):
            parse_structured_plan("here is my plan: fix the bug")
        with pytest.raises(PlanValidationError):
            parse_structured_plan({"problem": "x", "hypothesis": "y"})
        with pytest.raises(PlanValidationError):
            parse_structured_plan(
                {
                    "problem": "x",
                    "hypothesis": "y",
                    "files_to_inspect": "auth.py",
                    "steps": ["a", "b", "c"],
                }
            )
        with pytest.raises(PlanValidationError):
            parse_structured_plan(
                {
                    **VALID_PLAN,
                    "extra_key": "nope",
                }
            )
        with pytest.raises(PlanValidationError):
            parse_structured_plan(
                {
                    "problem": "x",
                    "hypothesis": "y",
                    "files_to_inspect": [],
                    "steps": ["only one"],
                }
            )
        with pytest.raises(PlanValidationError):
            parse_structured_plan(
                {
                    "problem": "x",
                    "hypothesis": "y",
                    "files_to_inspect": [],
                    "steps": ["a", "b", "c", "d", "e", "f"],
                }
            )

    def test_plan_node_rejects_malformed_llm_output(self) -> None:
        client = FakeClient([_Response("sorry, I cannot produce JSON")])
        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            pytest.raises(PlanValidationError),
        ):
            structured_plan(
                initial_state("bug"),
                _config(client),
            )

    def test_plan_prompt_includes_repo_inventory(self) -> None:
        client = FakeClient([_Response(json.dumps(VALID_PLAN))])
        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files) as list_mock,
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: path == "app/auth.py"),
            ),
        ):
            result = structured_plan(initial_state("bug"), _config(client))

        user = client.calls[0]["messages"][1]["content"]
        assert "Repository inventory" in user
        assert "app/auth.py" in user
        assert list_mock.call_count >= 1
        assert result["plan"]["files_to_inspect"] == ["app/auth.py"]

    def test_plan_filters_nonexistent_files(self) -> None:
        bad_plan = {
            **VALID_PLAN,
            "files_to_inspect": [
                "app/auth.py",
                "src/main/java/JwtFilter.java",
            ],
        }
        client = FakeClient([_Response(json.dumps(bad_plan))])
        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(
                    exists=lambda p=path: p == "app/auth.py"
                ),
            ),
        ):
            result = structured_plan(initial_state("bug"), _config(client))

        assert result["plan"]["files_to_inspect"] == ["app/auth.py"]


    def test_plan_prompt_prefers_retrieved_snippets(self) -> None:
        client = FakeClient([_Response(json.dumps(VALID_PLAN))])
        state = initial_state("bug")
        state["relevant_files"] = ["app/inventory.py"]
        state["retrieved_context"] = (
            "### app/inventory.py  allocate_bin\ndef allocate_bin():\n    return 1"
        )
        with (
            patch("agent.nodes.plan.list_files") as list_mock,
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
        ):
            structured_plan(state, _config(client))

        user = client.calls[0]["messages"][1]["content"]
        assert "Retrieved code" in user
        assert "allocate_bin" in user
        assert "Repository inventory" not in user
        list_mock.assert_not_called()


class TestExecuteContext:
    def test_workflow_context_is_plan_only(self) -> None:
        state = initial_state("bug")
        state["analysis"] = "long analysis that must not be repeated"
        state["plan"] = VALID_PLAN
        context = _workflow_context(state)
        assert "long analysis" not in context
        assert '"plan"' in context
        assert "Ignore plan paths" in context
        payload = json.loads(context)
        assert set(payload) == {"plan", "guardrail"}

    def test_workflow_context_includes_retrieved_snippets(self) -> None:
        state = initial_state("bug")
        state["plan"] = VALID_PLAN
        state["relevant_files"] = ["app/inventory.py"]
        state["retrieved_context"] = "### app/inventory.py  allocate_bin"
        payload = json.loads(_workflow_context(state))
        assert payload["retrieved"] == "### app/inventory.py  allocate_bin"
        assert payload["relevant_files"] == ["app/inventory.py"]
        assert "plan" in payload

    def test_workflow_context_includes_diagnosis_on_retry(self) -> None:
        state = initial_state("bug")
        state["plan"] = VALID_PLAN
        state["diagnosis"] = "Need to catch IndexError."
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
                "root_cause": "Need to catch IndexError.",
            }
        ]
        payload = json.loads(_workflow_context(state))
        assert payload["diagnosis"] == "Need to catch IndexError."
        assert payload["retry_count"] == 1
        assert payload["structured_diagnosis"]["new_hypothesis"] == (
            VALID_DIAGNOSIS["new_hypothesis"]
        )
        assert payload["attempt_history"][0]["attempt_index"] == 0

    def test_workflow_context_includes_human_feedback(self) -> None:
        state = initial_state("bug")
        state["plan"] = VALID_PLAN
        state["human_feedback"] = "Catch ExpiredSignatureError"
        state["human_retry_count"] = 1
        payload = json.loads(_workflow_context(state))
        assert payload["human_feedback"] == "Catch ExpiredSignatureError"
        assert payload["human_retry_count"] == 1


class TestRouting:
    def test_pass_and_fail_from_deterministic_pass(self) -> None:
        assert (
            route_after_verify({"test_result": {"deterministic_pass": True}})
            == "evaluate"
        )
        assert (
            route_after_verify({"test_result": {"deterministic_pass": False}})
            == "diagnose"
        )
        assert route_after_verify({}) == "diagnose"
        assert (
            route_after_evaluate(
                {
                    "test_result": {"deterministic_pass": True},
                    "patch_evaluation": VALID_EVALUATION,
                }
            )
            == "await_approval"
        )
        assert (
            route_after_evaluate(
                {
                    "test_result": {"deterministic_pass": True},
                    "patch_evaluation": {**VALID_EVALUATION, "issue_resolved": False},
                }
            )
            == "diagnose"
        )

    def test_approval_gate_routes(self) -> None:
        assert route_after_approval({}) == "mark_success"
        assert route_after_approval({"approval_decision": "approve"}) == "mark_success"
        assert (
            route_after_approval({"approval_decision": "reject"}) == "mark_needs_human"
        )
        assert route_after_approval({"approval_decision": "feedback"}) == "diagnose"

    def test_pytest_exit_code_alone_does_not_pass(self) -> None:
        assert (
            route_after_verify({"test_result": {"exit_code": 0, "passed": True}})
            == "diagnose"
        )

    def test_route_ignores_model_text_claims(self) -> None:
        state = {
            "analysis": "Tests appear to have passed.",
            "test_result": {
                "exit_code": 0,
                "passed": False,
                "deterministic_pass": False,
                "output": "exit_code=0\nAll tests passed",
            },
        }
        assert route_after_verify(state) == "diagnose"

    def test_diagnose_retries_until_budget(self) -> None:
        assert route_after_diagnose({"retry_count": 0}) == "plan"
        assert route_after_diagnose({"retry_count": 1}) == "plan"
        assert route_after_diagnose({"retry_count": 2}) == "feedback"
        assert route_after_feedback({"status": "feedback_retry"}) == "plan"
        assert route_after_feedback({"status": "feedback_skipped"}) == "mark_needs_human"
        assert route_after_feedback({"status": "feedback_declined"}) == "mark_needs_human"
        assert route_after_feedback({"status": "feedback_refused"}) == "mark_needs_human"


class TestWorkflowGraph:
    def test_pass_routes_to_success_without_diagnose(self) -> None:
        client = FakeClient(
            [
                _Response("Problem: expired JWT. Hypothesis: missing catch."),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
            ]
        )
        executor = {
            "final_answer": "fixed",
            "termination": "completed",
            "steps": 3,
            "tool_call_count": 4,
            "file_reads": 2,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "tokens": 120,
            "latency": 0.5,
            "trajectory": [{"tool": "read_file"}],
            "messages": [{"role": "user", "content": "hi"}],
        }

        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch("agent.nodes.execute.run_agent", return_value=executor) as run_agent_mock,
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_layer1_run_command(),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                return_value=PASS_PATCH,
            ),
            patch("agent.graph.diagnose_failure") as diagnose_mock,
        ):
            result = run_workflow(
                client=client,
                issue="Expired JWT returns 500",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                model="fake-model",
                max_steps=5,
                graph=build_graph(),
            )

        assert result["status"] == "success"
        assert result["workflow_passed"] is True
        assert result["termination"] == "completed"
        assert result["plan"]["problem"] == VALID_PLAN["problem"]
        assert result["analysis"]
        assert result["diagnosis"] == ""
        assert result["final_answer"] == "fixed"
        assert result["tool_call_count"] == 4
        assert result["file_reads"] == 2
        # analyze + plan + executor steps + evaluate
        assert result["llm_calls"] == 2 + 3 + 1
        assert result["tokens"] == 10 + 5 + 10 + 5 + 100 + 20 + 10 + 5
        assert "analyze" in result["stage_tokens"]
        assert "plan" in result["stage_tokens"]
        assert "execute" in result["stage_tokens"]
        assert "evaluate" in result["stage_tokens"]
        assert result["stage_tokens"]["execute"]["llm_calls"] == 3
        assert result["stage_tokens"]["evaluate"]["llm_calls"] == 1
        assert result["patch_evaluation"]["issue_resolved"] is True
        run_agent_mock.assert_called_once()
        exec_kwargs = run_agent_mock.call_args.kwargs
        assert exec_kwargs.get("search_code_enabled") is False
        exec_tools = [t["function"]["name"] for t in exec_kwargs["tools"]]
        assert "search_code" not in exec_tools
        payload = json.loads(exec_kwargs["workflow_context"])
        assert "plan" in payload
        assert "analysis" not in payload
        assert "guardrail" in payload
        diagnose_mock.assert_not_called()
        assert result["attempt_history"] == []
        assert result["retry_count"] == 0
        assert len(client.calls) == 3  # analyze + plan + evaluate
        assert "Repository inventory" in client.calls[1]["messages"][1]["content"]
        assert "Git diff" in client.calls[2]["messages"][1]["content"]

    def test_fail_retries_then_needs_human(self) -> None:
        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
                _Response(json.dumps(REVISED_PLAN)),
                _Response(json.dumps(STILL_FAILING_DIAGNOSIS)),
            ]
        )
        executor = {
            "final_answer": "attempted fix",
            "termination": "completed",
            "steps": 2,
            "tool_call_count": 3,
            "file_reads": 1,
            "prompt_tokens": 50,
            "completion_tokens": 10,
            "tokens": 60,
            "latency": 0.2,
            "trajectory": [{"tool": "edit_file"}],
            "messages": [],
        }

        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch("agent.nodes.execute.run_agent", return_value=executor) as run_agent_mock,
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_layer1_run_command(pytest_ok=False),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                return_value=EMPTY_PATCH,
            ),
        ):
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
            )

        assert result["status"] == "needs_human"
        assert result["workflow_passed"] is False
        assert result["termination"] == "needs_human"
        assert result["retry_count"] == 2
        assert result["human_retry_count"] == 0
        assert result["human_feedback"] == ""
        assert "Still failing" in result["diagnosis"]
        assert result["test_result"]["exit_code"] == 1
        assert run_agent_mock.call_count == 2
        assert len(result["attempt_history"]) == 2
        assert all(
            item["failure_source"] == "deterministic" for item in result["attempt_history"]
        )
        assert all(item["evaluator_pass"] is None for item in result["attempt_history"])
        retry_ctx = json.loads(run_agent_mock.call_args.kwargs["workflow_context"])
        assert "diagnosis" in retry_ctx
        assert "structured_diagnosis" in retry_ctx
        assert "attempt_history" in retry_ctx
        assert retry_ctx["plan"]["hypothesis"] == REVISED_PLAN["hypothesis"]
        assert retry_ctx["plan"]["hypothesis"] != VALID_PLAN["hypothesis"]
        assert len(client.calls) == 5
        assert result["llm_calls"] == 2 + 4 + 2 + 1
        assert "diagnose" in result["stage_tokens"]
        assert result["tool_call_count"] == 6
        assert len(result["trajectory"]) == 2

    def test_retry_then_pass(self) -> None:
        """Phase 5 trajectory: FAIL → diagnose → new hypothesis → retry → Layer1+Layer2 PASS."""
        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(RETRY_DIAGNOSIS)),
                _Response(json.dumps(RETRY_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
            ]
        )
        failed = {
            "final_answer": "not yet",
            "termination": "completed",
            "steps": 1,
            "tool_call_count": 2,
            "file_reads": 1,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "tokens": 12,
            "latency": 0.1,
            "trajectory": [{"tool": "edit_file"}],
            "messages": [],
        }
        passed = {
            **failed,
            "final_answer": "fixed",
            "trajectory": [{"tool": "run_tests"}],
        }

        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch(
                "agent.nodes.execute.run_agent",
                side_effect=[failed, passed],
            ),
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_fail_then_pass_commands(),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                side_effect=[EMPTY_PATCH, PASS_PATCH],
            ),
        ):
            reporter = RecordingReporter()
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
                progress=reporter,
            )

        assert result["status"] == "success"
        assert result["workflow_passed"] is True
        assert result["retry_count"] == 1
        assert result["test_result"]["deterministic_pass"] is True
        assert patch_evaluation_passed(result["patch_evaluation"]) is True
        assert "409" in result["diagnosis"]
        assert result["plan"]["hypothesis"] == RETRY_PLAN["hypothesis"]
        assert result["plan"]["hypothesis"] != VALID_PLAN["hypothesis"]
        assert result["final_answer"] == "fixed"
        assert len(result["attempt_history"]) == 1
        assert result["attempt_history"][0]["failure_source"] == "deterministic"
        assert result["attempt_history"][0]["evaluator_pass"] is None
        stages = [(e[1], e[2]) for e in reporter.events if e[0] == "stage"]
        assert stages[0][0] == "analyze"
        assert ("execute", "") in stages
        assert ("verify", "FAIL  pytest=0 ruff=1 patch=0") in stages
        assert stages[[s[0] for s in stages].index("diagnose")][1].startswith("Retry")
        assert any(s[0] == "plan" and str(s[1]).startswith("retry") for s in stages)
        assert ("execute", "retry 1") in stages
        assert ("verify", "PASS  pytest=1 ruff=1 patch=1") in stages
        assert any(s[0] == "evaluate" and str(s[1]).startswith("PASS") for s in stages)
        assert ("success", "") in stages
        assert result["stage_tokens"]["evaluate"]["llm_calls"] == 1

    def test_evaluator_reject_then_retry_pass(self) -> None:
        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(REJECT_EVALUATION)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
                _Response(json.dumps(REVISED_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
            ]
        )
        first = {
            "final_answer": "too narrow",
            "termination": "completed",
            "steps": 1,
            "tool_call_count": 2,
            "file_reads": 1,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "tokens": 12,
            "latency": 0.1,
            "trajectory": [{"tool": "edit_file"}],
            "messages": [],
        }
        second = {**first, "final_answer": "fixed"}

        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch(
                "agent.nodes.execute.run_agent",
                side_effect=[first, second],
            ) as run_agent_mock,
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_layer1_run_command(),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                return_value=PASS_PATCH,
            ),
        ):
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
            )

        assert result["status"] == "success"
        assert result["workflow_passed"] is True
        assert result["retry_count"] == 1
        assert result["plan"]["hypothesis"] == REVISED_PLAN["hypothesis"]
        assert result["plan"]["hypothesis"] != VALID_PLAN["hypothesis"]
        assert len(result["attempt_history"]) == 1
        assert result["attempt_history"][0]["failure_source"] == "evaluator"
        assert result["attempt_history"][0]["evaluator_pass"] is False
        assert run_agent_mock.call_count == 2
        retry_ctx = json.loads(run_agent_mock.call_args.kwargs["workflow_context"])
        assert "diagnosis" in retry_ctx
        assert retry_ctx["plan"]["hypothesis"] == REVISED_PLAN["hypothesis"]
        assert result["patch_evaluation"]["issue_resolved"] is True

    def test_evaluator_reject_then_needs_human(self) -> None:
        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(REJECT_EVALUATION)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
                _Response(json.dumps(REVISED_PLAN)),
                _Response(json.dumps(REJECT_EVALUATION)),
                _Response(json.dumps(STILL_FAILING_DIAGNOSIS)),
            ]
        )
        executor = {
            "final_answer": "attempted fix",
            "termination": "completed",
            "steps": 1,
            "tool_call_count": 2,
            "file_reads": 1,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "tokens": 12,
            "latency": 0.1,
            "trajectory": [{"tool": "edit_file"}],
            "messages": [],
        }

        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch("agent.nodes.execute.run_agent", return_value=executor) as run_agent_mock,
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_layer1_run_command(),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                return_value=PASS_PATCH,
            ),
        ):
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
            )

        assert result["status"] == "needs_human"
        assert result["workflow_passed"] is False
        assert result["termination"] == "needs_human"
        assert result["retry_count"] == 2
        assert result["human_retry_count"] == 0
        assert run_agent_mock.call_count == 2
        assert len(result["attempt_history"]) == 2
        assert all(
            item["failure_source"] == "evaluator" for item in result["attempt_history"]
        )
        assert all(item["evaluator_pass"] is False for item in result["attempt_history"])
        assert "Still failing" in result["diagnosis"]

    def test_verify_pass_despite_model_claiming_failure_text(self) -> None:
        client = FakeClient(
            [
                _Response("I believe the tests failed."),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
            ]
        )
        executor = {
            "final_answer": "done",
            "termination": "completed",
            "steps": 1,
            "tool_call_count": 0,
            "file_reads": 0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "tokens": 2,
            "latency": 0.1,
            "trajectory": [],
            "messages": [],
        }

        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch("agent.nodes.execute.run_agent", return_value=executor),
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_layer1_run_command(),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                return_value=PASS_PATCH,
            ),
            patch("agent.graph.diagnose_failure") as diagnose_mock,
        ):
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
            )

        assert result["workflow_passed"] is True
        assert result["status"] == "success"
        diagnose_mock.assert_not_called()


class ScriptedProvider:
    def __init__(self, replies: list[str | None]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("unexpected feedback prompt")
        return self.replies.pop(0)


def _short_executor(answer: str = "attempted") -> dict[str, Any]:
    return {
        "final_answer": answer,
        "termination": "completed",
        "steps": 1,
        "tool_call_count": 2,
        "file_reads": 1,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "tokens": 12,
        "latency": 0.1,
        "trajectory": [{"tool": "edit_file"}],
        "messages": [],
    }


class TestFeedbackRecovery:
    def test_declined_feedback_escalates(self) -> None:
        provider = ScriptedProvider([""])
        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
                _Response(json.dumps(REVISED_PLAN)),
                _Response(json.dumps(STILL_FAILING_DIAGNOSIS)),
            ]
        )
        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch("agent.nodes.execute.run_agent", return_value=_short_executor()),
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_layer1_run_command(pytest_ok=False),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                return_value=EMPTY_PATCH,
            ),
        ):
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
                feedback_provider=provider,
            )

        assert result["status"] == "needs_human"
        assert result["workflow_passed"] is False
        assert result["human_retry_count"] == 0
        assert result["human_feedback"] == ""
        assert len(provider.prompts) == 1
        assert "Automatic retry budget exhausted" in provider.prompts[0]

    def test_feedback_retry_then_pass(self) -> None:
        provider = ScriptedProvider([HUMAN_FEEDBACK])
        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
                _Response(json.dumps(REVISED_PLAN)),
                _Response(json.dumps(STILL_FAILING_DIAGNOSIS)),
                _Response(json.dumps(FEEDBACK_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
            ]
        )
        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch(
                "agent.nodes.execute.run_agent",
                side_effect=[
                    _short_executor("not yet"),
                    _short_executor("still failing"),
                    _short_executor("fixed"),
                ],
            ) as run_agent_mock,
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_fail_n_then_pass_commands(2),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                side_effect=[EMPTY_PATCH, EMPTY_PATCH, PASS_PATCH],
            ),
        ):
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
                feedback_provider=provider,
            )

        assert result["status"] == "success"
        assert result["workflow_passed"] is True
        assert result["human_retry_count"] == 1
        assert result["human_feedback"] == HUMAN_FEEDBACK
        assert result["plan"]["hypothesis"] == FEEDBACK_PLAN["hypothesis"]
        assert run_agent_mock.call_count == 3
        retry_ctx = json.loads(run_agent_mock.call_args.kwargs["workflow_context"])
        assert retry_ctx["human_feedback"] == HUMAN_FEEDBACK
        plan_user = client.calls[5]["messages"][1]["content"]
        assert "Human feedback" in plan_user
        assert HUMAN_FEEDBACK in plan_user

    def test_second_feedback_attempt_is_refused(self) -> None:
        provider = ScriptedProvider([HUMAN_FEEDBACK, "should not be asked"])
        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
                _Response(json.dumps(REVISED_PLAN)),
                _Response(json.dumps(STILL_FAILING_DIAGNOSIS)),
                _Response(json.dumps(FEEDBACK_PLAN)),
                _Response(json.dumps(STILL_FAILING_DIAGNOSIS)),
            ]
        )
        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch(
                "agent.nodes.execute.run_agent",
                return_value=_short_executor(),
            ) as run_agent_mock,
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_layer1_run_command(pytest_ok=False),
            ),
            patch(
                "agent.nodes.verify.inspect_worktree",
                return_value=EMPTY_PATCH,
            ),
        ):
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
                feedback_provider=provider,
            )

        assert result["status"] == "needs_human"
        assert result["workflow_passed"] is False
        assert result["human_retry_count"] == 1
        assert run_agent_mock.call_count == 3
        assert len(provider.prompts) == 1
        assert provider.replies == ["should not be asked"]


class TestSandboxRuntimeConfig:
    def test_sandbox_is_threaded_not_serialized_in_state(self) -> None:
        fake_sandbox = object()
        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
            ]
        )
        executor = {
            "final_answer": "fixed",
            "termination": "completed",
            "steps": 1,
            "tool_call_count": 0,
            "file_reads": 0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "tokens": 2,
            "latency": 0.1,
            "trajectory": [],
            "messages": [],
        }

        with (
            patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
            patch("agent.nodes.execute.run_agent", return_value=executor) as run_agent_mock,
            patch(
                "agent.nodes.verify.run_command",
                side_effect=_layer1_run_command(),
            ) as run_command_mock,
            patch(
                "agent.nodes.verify.inspect_worktree",
                return_value=PASS_PATCH,
            ) as inspect_mock,
            patch("agent.graph.diagnose_failure") as diagnose_mock,
        ):
            result = run_workflow(
                client=client,
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                sandbox=fake_sandbox,
                graph=build_graph(),
            )

        assert result["workflow_passed"] is True
        assert run_agent_mock.call_args.kwargs["sandbox"] is fake_sandbox
        assert run_command_mock.call_args.kwargs["sandbox"] is fake_sandbox
        assert inspect_mock.call_args.kwargs["sandbox"] is fake_sandbox
        diagnose_mock.assert_not_called()
        assert "sandbox" not in (result.get("messages") or [])
        for key in ("final_answer", "termination", "steps", "tokens", "trajectory"):
            assert key in result
        assert "sandbox" not in result


class TestV0Seam:
    def test_default_prompt_unchanged_without_context(self) -> None:
        client = FakeClient([_Response("all done")])
        result = run_agent(
            client=client,
            issue="Expired JWT returns 500",
            repo_path="/tmp/repo",
            test_command="pytest -q",
            model="fake-model",
            max_steps=3,
        )
        user = client.calls[0]["messages"][1]["content"]
        assert user == "Fix this issue:\n\nExpired JWT returns 500"
        assert "Workflow context" not in user
        assert set(result) >= {
            "final_answer",
            "termination",
            "steps",
            "tool_call_count",
            "file_reads",
            "tokens",
            "trajectory",
            "messages",
        }
        assert result["final_answer"] == "all done"
        assert "tools" in client.calls[0]
        tool_names = [t["function"]["name"] for t in client.calls[0]["tools"]]
        assert "search_code" not in tool_names
        assert len(tool_names) == 6

    def test_workflow_context_appended_when_provided(self) -> None:
        client = FakeClient([_Response("done")])
        run_agent(
            client=client,
            issue="bug",
            repo_path="/tmp/repo",
            test_command="pytest -q",
            workflow_context='{"plan": {"steps": ["a"]}}',
        )
        user = client.calls[0]["messages"][1]["content"]
        assert "Workflow context:" in user
        assert "steps" in user


class TestAdaptResult:
    def test_maps_common_keys(self) -> None:
        state = initial_state("bug")
        state["status"] = "success"
        state["analysis"] = "a"
        state["plan"] = VALID_PLAN
        state["test_result"] = {
            "exit_code": 0,
            "passed": True,
            "deterministic_pass": True,
        }
        state["patch_evaluation"] = VALID_EVALUATION
        state["telemetry"] = {
            "tool_call_count": 2,
            "file_reads": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "tokens": 15,
            "steps": 3,
            "llm_calls": 4,
            "final_answer": "fixed",
            "trajectory": [],
            "stage_tokens": {"analyze": {"prompt_tokens": 10, "completion_tokens": 5, "llm_calls": 1}},
        }
        out = adapt_result(state)
        assert out["termination"] == "completed"
        assert out["workflow_passed"] is True
        assert out["tokens"] == 15
        assert out["plan"] == VALID_PLAN
        assert out["retry_count"] == 0
        assert out["stage_tokens"]["analyze"]["llm_calls"] == 1
        assert out["patch_evaluation"]["issue_resolved"] is True

    def test_needs_human_termination(self) -> None:
        state = initial_state("bug")
        state["status"] = "needs_human"
        state["retry_count"] = 2
        state["test_result"] = {"exit_code": 1, "passed": False}
        out = adapt_result(state)
        assert out["termination"] == "needs_human"
        assert out["retry_count"] == 2
        assert out["workflow_passed"] is False
