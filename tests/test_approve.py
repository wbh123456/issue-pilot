"""Approval gate: interrupt payload, routing, and checkpointer requirement."""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langgraph.types import Command

from agent.graph import (
    build_graph,
    build_runtime_config,
    route_after_approval,
    route_after_diagnose,
    route_after_feedback,
    run_workflow,
)
from agent.nodes.approve import (
    ApprovalError,
    await_approval,
    interrupt_payload,
    parse_approval_resume,
    review_payload,
)
from agent.state import initial_state
from harness.checkpoint import open_checkpointer
from harness.limits import MAX_RETRY
from tests.test_checkpoint import _completed_nodes, _pass_patches
from tests.test_graph import (
    EMPTY_PATCH,
    FakeClient,
    HUMAN_FEEDBACK,
    PASS_PATCH,
    REVISED_PLAN,
    STILL_FAILING_DIAGNOSIS,
    VALID_DIAGNOSIS,
    VALID_EVALUATION,
    VALID_PLAN,
    _Response,
    _fail_n_then_pass_commands,
    _fake_list_files,
    _short_executor,
)


def _pass_client() -> FakeClient:
    return FakeClient(
        [
            _Response("Problem: expired JWT. Hypothesis: missing catch."),
            _Response(json.dumps(VALID_PLAN)),
            _Response(json.dumps(VALID_EVALUATION)),
        ]
    )


def _interrupt_pass(
    saver,
    client: FakeClient,
    *,
    thread_id: str = "issue-001-v1-approve",
    feedback_provider=None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    compiled = build_graph(checkpointer=saver)
    config = build_runtime_config(
        client=client,
        repo_path="/tmp/repo",
        test_command="pytest -q",
        model="fake-model",
        max_steps=5,
        thread_id=thread_id,
        require_approval=True,
        feedback_provider=feedback_provider,
    )
    with ExitStack() as stack:
        for patcher in _pass_patches():
            stack.enter_context(patcher)
        result = compiled.invoke(initial_state("Expired JWT returns 500"), config)
    return compiled, result, config


class TestReviewContract:
    def test_payload_has_six_cli_fields(self) -> None:
        state = initial_state("Expired JWT returns 500")
        state["plan"] = VALID_PLAN
        state["patch_evaluation"] = VALID_EVALUATION
        state["test_result"] = {
            "deterministic_pass": True,
            "pytest_passed": True,
            "ruff_passed": True,
            "patch_valid": True,
            "exit_code": 0,
            "output": "1 passed",
            "changed_files": ["app/auth.py"],
            "git_diff": "diff --git a/app/auth.py b/app/auth.py\n+fixed\n",
        }
        payload = review_payload(state)
        assert set(payload) == {
            "issue",
            "plan",
            "changed_files",
            "git_diff",
            "test_result",
            "evaluator_result",
        }
        assert payload["issue"] == "Expired JWT returns 500"
        assert payload["plan"]["problem"] == VALID_PLAN["problem"]
        assert payload["changed_files"] == ["app/auth.py"]
        assert "app/auth.py" in payload["git_diff"]
        assert payload["test_result"]["deterministic_pass"] is True
        assert payload["evaluator_result"]["issue_resolved"] is True

    def test_parse_resume_fail_closed(self) -> None:
        assert parse_approval_resume("approve") == ("approve", "")
        assert parse_approval_resume({"decision": "reject"}) == ("reject", "")
        assert parse_approval_resume(
            {"decision": "feedback", "feedback": "tighten the patch"}
        ) == ("feedback", "tighten the patch")
        assert parse_approval_resume("nope") == ("reject", "")
        assert parse_approval_resume({"decision": "feedback"}) == ("reject", "")
        assert parse_approval_resume(None) == ("reject", "")


class TestAwaitApprovalNode:
    def test_gate_off_is_noop(self) -> None:
        state = initial_state("bug")
        assert await_approval(state, {"configurable": {}}) == {}
        assert await_approval(state, {"configurable": {"require_approval": False}}) == {}

    def test_gate_on_without_checkpointer_raises(self) -> None:
        with pytest.raises(ApprovalError, match="checkpointer"):
            await_approval(
                initial_state("bug"),
                {"configurable": {"require_approval": True}},
            )


class TestApprovalRouting:
    def test_decisions(self) -> None:
        assert route_after_approval({}) == "mark_success"
        assert route_after_approval({"approval_decision": "approve"}) == "mark_success"
        assert (
            route_after_approval({"approval_decision": "reject"}) == "mark_needs_human"
        )
        assert route_after_approval({"approval_decision": "feedback"}) == "diagnose"

    def test_budget_exhausted_feedback_escalates(self) -> None:
        assert route_after_approval({"approval_decision": "feedback"}) == "diagnose"
        assert route_after_diagnose({"retry_count": MAX_RETRY}) == "feedback"
        assert route_after_feedback({"status": "feedback_refused"}) == "mark_needs_human"


class TestApprovalGraph:
    def test_require_approval_without_checkpointer_raises(self) -> None:
        with pytest.raises(ApprovalError, match="checkpointer"):
            run_workflow(
                client=FakeClient([]),
                issue="bug",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(),
                require_approval=True,
            )

    def test_gate_off_still_succeeds_without_checkpointer(self) -> None:
        client = _pass_client()
        with ExitStack() as stack:
            for patcher in _pass_patches():
                stack.enter_context(patcher)
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
        assert result["approval_decision"] == ""

    def test_gate_on_interrupts_with_review_payload(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        with open_checkpointer(db) as saver:
            compiled, result, config = _interrupt_pass(saver, _pass_client())
            payload = interrupt_payload(result)
            assert payload is not None
            assert set(payload) >= {
                "issue",
                "plan",
                "changed_files",
                "git_diff",
                "test_result",
                "evaluator_result",
            }
            assert payload["issue"] == "Expired JWT returns 500"
            assert payload["plan"]["problem"] == VALID_PLAN["problem"]
            assert payload["changed_files"] == ["app/auth.py"]
            assert payload["evaluator_result"]["issue_resolved"] is True
            assert compiled.get_state(config).next == ("await_approval",)

        with open_checkpointer(db) as saver, ExitStack() as stack:
            for patcher in _pass_patches():
                stack.enter_context(patcher)
            result = run_workflow(
                client=_pass_client(),
                issue="Expired JWT returns 500",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                model="fake-model",
                max_steps=5,
                graph=build_graph(checkpointer=saver),
                thread_id="run-workflow-pause",
                require_approval=True,
            )
        assert result["status"] == "waiting_approval"
        assert result["workflow_passed"] is False
        assert result["approval_payload"]["plan"]["problem"] == VALID_PLAN["problem"]

    def test_approve_then_success(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        with open_checkpointer(db) as saver:
            compiled, result, config = _interrupt_pass(saver, _pass_client())
            assert interrupt_payload(result)
            with ExitStack() as stack:
                for patcher in _pass_patches():
                    stack.enter_context(patcher)
                final = compiled.invoke(Command(resume="approve"), config)
            assert final["status"] == "success"
            assert final["approval_decision"] == "approve"
            assert _completed_nodes(compiled, config)[-2:] == [
                "await_approval",
                "mark_success",
            ]

    def test_reject_then_needs_human(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        with open_checkpointer(db) as saver:
            compiled, result, config = _interrupt_pass(saver, _pass_client())
            assert interrupt_payload(result)
            with ExitStack() as stack:
                for patcher in _pass_patches():
                    stack.enter_context(patcher)
                final = compiled.invoke(Command(resume="reject"), config)
            assert final["status"] == "needs_human"
            assert final["approval_decision"] == "reject"
            assert "mark_needs_human" in _completed_nodes(compiled, config)

    def test_feedback_then_diagnose(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        client = FakeClient(
            [
                _Response("Problem: expired JWT. Hypothesis: missing catch."),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
                _Response(json.dumps(REVISED_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
            ]
        )
        with open_checkpointer(db) as saver:
            compiled, paused, config = _interrupt_pass(
                saver, client, thread_id="feedback-route"
            )
            assert interrupt_payload(paused)
            with ExitStack() as stack:
                for patcher in _pass_patches():
                    stack.enter_context(patcher)
                resumed = compiled.invoke(
                    Command(
                        resume={
                            "decision": "feedback",
                            "feedback": "Drop the unrelated auth.py edit",
                        }
                    ),
                    config,
                )
            assert interrupt_payload(resumed) is not None
            assert compiled.get_state(config).values["approval_decision"] == "feedback"
            assert (
                compiled.get_state(config).values["human_feedback"]
                == "Drop the unrelated auth.py edit"
            )
            assert "diagnose" in _completed_nodes(compiled, config)
            diagnose_user = client.calls[3]["messages"][1]["content"]
            assert "Reviewer feedback" in diagnose_user
            assert "Drop the unrelated auth.py edit" in diagnose_user

    def test_budget_exhausted_approval_feedback_escalates(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        provider_calls: list[str] = []

        def provider(prompt: str) -> str:
            provider_calls.append(prompt)
            return HUMAN_FEEDBACK

        client = FakeClient(
            [
                _Response("analysis text"),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
                _Response(json.dumps(REVISED_PLAN)),
                _Response(json.dumps(STILL_FAILING_DIAGNOSIS)),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
                _Response(json.dumps(VALID_DIAGNOSIS)),
            ]
        )
        with open_checkpointer(db) as saver:
            compiled = build_graph(checkpointer=saver)
            config = build_runtime_config(
                client=client,
                repo_path="/tmp/repo",
                test_command="pytest -q",
                model="fake-model",
                max_steps=5,
                thread_id="budget-exhausted",
                require_approval=True,
                feedback_provider=provider,
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
                ),
                patch(
                    "agent.nodes.verify.run_command",
                    side_effect=_fail_n_then_pass_commands(2),
                ),
                patch(
                    "agent.nodes.verify.inspect_worktree",
                    side_effect=[EMPTY_PATCH, EMPTY_PATCH, PASS_PATCH],
                ),
            ):
                paused = compiled.invoke(initial_state("bug"), config)
                assert interrupt_payload(paused)
                assert int(compiled.get_state(config).values.get("retry_count") or 0) == 2
                final = compiled.invoke(
                    Command(
                        resume={
                            "decision": "feedback",
                            "feedback": "Drop the unrelated files",
                        }
                    ),
                    config,
                )
            assert final["status"] == "needs_human"
            assert final["approval_decision"] == "feedback"
            assert len(provider_calls) == 1
            nodes = _completed_nodes(compiled, config)
            assert "diagnose" in nodes
            assert "mark_needs_human" in nodes
