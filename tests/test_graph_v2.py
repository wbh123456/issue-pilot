"""V2 graph: analyze → retrieve → plan. V1 singleton stays retrieve-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableConfig

from agent.graph import (
    adapt_result,
    build_graph,
    get_graph,
    get_v2_graph,
    run_workflow,
)
from agent.nodes.execute import execute_plan
from agent.nodes.retrieve import retrieve_context
from agent.state import initial_state
from agent.tools.git import WorktreeDiff
from agent.tools.shell import CommandOutcome
from retrieval.dense import HashingEmbedder


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

REVISED_PLAN = {
    **VALID_PLAN,
    "hypothesis": "Need to catch ExpiredSignatureError",
    "steps": ["inspect auth.py", "catch expired JWT", "re-run tests"],
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
    "feedback": "missed allocate_bin overflow",
}

_FAKE_INVENTORY = {
    ".": "app/\ntests/\nREADME.md",
    "app": "app/auth.py\napp/main.py",
    "tests": "tests/test_auth.py",
}


def _fake_list_files(repo_path: str, path: str = ".") -> str:
    return _FAKE_INVENTORY.get(path, "Error: path not found")


PASS_PATCH = WorktreeDiff(
    status=" M app/inventory.py",
    diff="diff --git a/app/inventory.py b/app/inventory.py\n+fixed\n",
)
EMPTY_PATCH = WorktreeDiff()


def _layer1_run_command(*, pytest_ok: bool = True, ruff_ok: bool = True):
    def _run(repo_path: str, command: str, **kwargs: Any) -> CommandOutcome:
        is_ruff = command.strip().startswith("ruff")
        ok = ruff_ok if is_ruff else pytest_ok
        return CommandOutcome(
            command=command.split(),
            exit_code=0 if ok else 1,
            stdout="ok" if ok else "FAILED",
        )

    return _run


def _node_ids(compiled) -> set[str]:
    nodes = compiled.get_graph().nodes
    if isinstance(nodes, dict):
        return set(nodes)
    return {str(getattr(node, "id", node)) for node in nodes}


def _edges(compiled) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for edge in compiled.get_graph().edges:
        src = getattr(edge, "source", None)
        tgt = getattr(edge, "target", None)
        if src is None and isinstance(edge, (tuple, list)) and len(edge) >= 2:
            src, tgt = edge[0], edge[1]
        pairs.add((str(src), str(tgt)))
    return pairs


def _fake_retrieve(state: dict, config: RunnableConfig) -> dict:
    telemetry = dict(state.get("telemetry") or {})
    telemetry["retrieval_calls"] = int(telemetry.get("retrieval_calls") or 0) + 1
    stages = dict(telemetry.get("stage_tokens") or {})
    stages["retrieve"] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_calls": 0,
    }
    telemetry["stage_tokens"] = stages
    return {
        "relevant_files": ["app/inventory.py", "app/orders.py"],
        "retrieved_context": "### app/inventory.py  allocate_bin\ndef allocate_bin():\n    pass",
        "status": "retrieved",
        "telemetry": telemetry,
    }


class TestGraphTopology:
    def test_v1_has_analyze_to_plan_without_retrieve(self) -> None:
        compiled = build_graph()
        nodes = _node_ids(compiled)
        edges = _edges(compiled)
        assert "retrieve" not in nodes
        assert ("analyze", "plan") in edges
        assert ("analyze", "retrieve") not in edges
        assert ("diagnose", "plan") in edges
        assert ("diagnose", "execute") not in edges
        assert "mark_needs_human" in nodes
        assert "feedback" in nodes
        assert ("diagnose", "feedback") in edges
        assert ("diagnose", "mark_needs_human") not in edges
        assert ("feedback", "plan") in edges
        assert ("feedback", "mark_needs_human") in edges
        assert "evaluate" in nodes
        assert "await_approval" in nodes
        assert ("verify", "evaluate") in edges
        assert ("evaluate", "await_approval") in edges
        assert ("evaluate", "mark_success") not in edges
        assert ("await_approval", "mark_success") in edges
        assert ("await_approval", "mark_needs_human") in edges
        assert ("await_approval", "diagnose") in edges
        assert ("evaluate", "diagnose") in edges
        assert ("verify", "mark_success") not in edges

    def test_v2_inserts_retrieve_between_analyze_and_plan(self) -> None:
        compiled = build_graph(include_retrieve=True)
        nodes = _node_ids(compiled)
        edges = _edges(compiled)
        assert "retrieve" in nodes
        assert ("analyze", "retrieve") in edges
        assert ("retrieve", "plan") in edges
        assert ("analyze", "plan") not in edges
        assert ("diagnose", "plan") in edges
        assert ("diagnose", "execute") not in edges
        assert "mark_needs_human" in nodes
        assert "feedback" in nodes
        assert ("diagnose", "feedback") in edges
        assert ("diagnose", "mark_needs_human") not in edges
        assert ("feedback", "plan") in edges
        assert ("feedback", "mark_needs_human") in edges
        assert "evaluate" in nodes
        assert "await_approval" in nodes
        assert ("verify", "evaluate") in edges
        assert ("evaluate", "await_approval") in edges
        assert ("evaluate", "mark_success") not in edges
        assert ("await_approval", "mark_success") in edges
        assert ("await_approval", "mark_needs_human") in edges
        assert ("await_approval", "diagnose") in edges
        assert ("verify", "mark_success") not in edges

    def test_singletons_are_distinct(self) -> None:
        v1 = get_graph()
        v2 = get_v2_graph()
        assert v1 is not v2
        assert "retrieve" not in _node_ids(v1)
        assert "retrieve" in _node_ids(v2)
        assert get_graph() is v1
        assert get_v2_graph() is v2


class TestRetrieveNode:
    def test_hybrid_search_zero_llm(self, tmp_path: Path) -> None:
        app = tmp_path / "app"
        app.mkdir()
        (app / "inventory.py").write_text(
            '"""Warehouse stock."""\n\n'
            "def allocate_bin(items):\n"
            '    return "A1"\n',
            encoding="utf-8",
        )
        (app / "validators.py").write_text(
            "def validate_email(value):\n    return True\n",
            encoding="utf-8",
        )
        state = initial_state("Ordering 50 widgets crashes with 500")
        state["analysis"] = "allocate_bin indexes slots by qty and can throw."
        config: dict[str, Any] = {
            "configurable": {
                "repo_path": str(tmp_path),
                "embedder": HashingEmbedder(),
            }
        }
        out = retrieve_context(state, config)
        assert out["status"] == "retrieved"
        assert "app/inventory.py" in out["relevant_files"]
        assert "allocate_bin" in out["retrieved_context"]
        assert out["telemetry"]["retrieval_calls"] == 1
        assert out["telemetry"]["stage_tokens"]["retrieve"]["llm_calls"] == 0
        assert out["telemetry"]["llm_calls"] == 0
        assert out["telemetry"]["query_mode"] == "issue"
        assert out["telemetry"]["retrieve_query"] == state["issue"]
        assert "indexes slots" not in out["telemetry"]["retrieve_query"]

    def test_issue_plus_analysis_query(self, tmp_path: Path) -> None:
        app = tmp_path / "app"
        app.mkdir()
        (app / "inventory.py").write_text(
            "def allocate_bin(items):\n    return 'A1'\n",
            encoding="utf-8",
        )
        state = initial_state("Ordering 50 widgets crashes with 500")
        state["analysis"] = "allocate_bin indexes slots by qty and can throw."
        config: dict[str, Any] = {
            "configurable": {
                "repo_path": str(tmp_path),
                "embedder": HashingEmbedder(),
                "embedder_name": "hashing",
                "query_mode": "issue+analysis",
            }
        }
        out = retrieve_context(state, config)
        query = out["telemetry"]["retrieve_query"]
        assert state["issue"] in query
        assert "allocate_bin" in query
        assert out["telemetry"]["query_mode"] == "issue+analysis"
        assert out["telemetry"]["embedder_name"] == "hashing"


class TestV2Workflow:
    def test_retrieve_runs_without_extra_llm_call(self) -> None:
        client = FakeClient(
            [
                _Response("Hypothesis: allocate_bin overflow."),
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

        retrieve_calls: list[int] = []

        def fake_retrieve(state: dict, config: RunnableConfig) -> dict:
            retrieve_calls.append(1)
            return _fake_retrieve(state, config)

        with (
            patch(
                "agent.nodes.retrieve.retrieve_context",
                new=fake_retrieve,
            ),
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
                issue="Ordering 50 widgets crashes with 500",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(include_retrieve=True),
            )

        assert retrieve_calls == [1]
        assert len(client.calls) == 3
        plan_user = client.calls[1]["messages"][1]["content"]
        assert "Retrieved code" in plan_user
        assert "allocate_bin" in plan_user
        assert "Repository inventory" not in plan_user
        payload = json.loads(run_agent_mock.call_args.kwargs["workflow_context"])
        assert "retrieved" in payload
        assert payload["relevant_files"] == ["app/inventory.py", "app/orders.py"]
        assert result["relevant_files"] == ["app/inventory.py", "app/orders.py"]
        assert result["retrieval_calls"] == 1
        assert result["stage_tokens"]["retrieve"]["llm_calls"] == 0
        assert result["workflow_passed"] is True
        assert result["llm_calls"] == 2 + 1 + 1
        assert result["stage_tokens"]["evaluate"]["llm_calls"] == 1
        assert run_agent_mock.call_args.kwargs.get("search_code_enabled") is False

    def test_fail_retries_then_needs_human_without_re_retrieve(self) -> None:
        client = FakeClient(
            [
                _Response("Hypothesis: allocate_bin overflow."),
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

        retrieve_calls: list[int] = []

        def fake_retrieve(state: dict, config: RunnableConfig) -> dict:
            retrieve_calls.append(1)
            return _fake_retrieve(state, config)

        with (
            patch(
                "agent.nodes.retrieve.retrieve_context",
                new=fake_retrieve,
            ),
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
                issue="Ordering 50 widgets crashes with 500",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(include_retrieve=True),
            )

        assert retrieve_calls == [1]
        assert result["status"] == "needs_human"
        assert result["workflow_passed"] is False
        assert result["termination"] == "needs_human"
        assert result["retry_count"] == 2
        assert result["human_retry_count"] == 0
        assert "Still failing" in result["diagnosis"]
        assert run_agent_mock.call_count == 2
        assert len(result["attempt_history"]) == 2
        assert all(
            item["failure_source"] == "deterministic" for item in result["attempt_history"]
        )
        retry_ctx = json.loads(run_agent_mock.call_args.kwargs["workflow_context"])
        assert "diagnosis" in retry_ctx
        assert "attempt_history" in retry_ctx
        assert retry_ctx["plan"]["hypothesis"] == REVISED_PLAN["hypothesis"]
        assert len(client.calls) == 5

    def test_evaluator_reject_retries_without_re_retrieve(self) -> None:
        client = FakeClient(
            [
                _Response("Hypothesis: allocate_bin overflow."),
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
            "tool_call_count": 0,
            "file_reads": 0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "tokens": 2,
            "latency": 0.1,
            "trajectory": [],
            "messages": [],
        }
        second = {**first, "final_answer": "fixed"}

        retrieve_calls: list[int] = []

        def fake_retrieve(state: dict, config: RunnableConfig) -> dict:
            retrieve_calls.append(1)
            return _fake_retrieve(state, config)

        with (
            patch(
                "agent.nodes.retrieve.retrieve_context",
                new=fake_retrieve,
            ),
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
                issue="Ordering 50 widgets crashes with 500",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                graph=build_graph(include_retrieve=True),
            )

        assert retrieve_calls == [1]
        assert result["status"] == "success"
        assert result["workflow_passed"] is True
        assert result["retry_count"] == 1
        assert result["attempt_history"][0]["failure_source"] == "evaluator"
        assert result["plan"]["hypothesis"] == REVISED_PLAN["hypothesis"]
        assert run_agent_mock.call_count == 2
        retry_ctx = json.loads(run_agent_mock.call_args.kwargs["workflow_context"])
        assert "diagnosis" in retry_ctx
        assert "retrieved" in retry_ctx


class TestExecuteSearchCode:
    def _executor_result(self, *, search_calls: int = 0) -> dict[str, Any]:
        trajectory = [
            {"tool": "search_code", "arguments": {"query": "greet"}}
            for _ in range(search_calls)
        ]
        return {
            "final_answer": "fixed",
            "termination": "completed",
            "steps": 1,
            "tool_call_count": search_calls,
            "file_reads": 0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "tokens": 2,
            "latency": 0.1,
            "trajectory": trajectory,
            "messages": [],
        }

    def test_disabled_keeps_six_tools(self) -> None:
        with patch(
            "agent.nodes.execute.run_agent",
            return_value=self._executor_result(),
        ) as mock:
            execute_plan(
                initial_state("bug"),
                {
                    "configurable": {
                        "client": object(),
                        "model": "m",
                        "repo_path": "/tmp/repo",
                        "test_command": "pytest -q",
                    }
                },
            )
        assert mock.call_args.kwargs["search_code_enabled"] is False
        names = [t["function"]["name"] for t in mock.call_args.kwargs["tools"]]
        assert "search_code" not in names

    def test_enabled_passes_v2_tools_and_counts_calls(self) -> None:
        state = initial_state("bug")
        state["telemetry"]["retrieval_calls"] = 1
        with patch(
            "agent.nodes.execute.run_agent",
            return_value=self._executor_result(search_calls=2),
        ) as mock:
            out = execute_plan(
                state,
                {
                    "configurable": {
                        "client": object(),
                        "model": "m",
                        "repo_path": "/tmp/repo",
                        "test_command": "pytest -q",
                        "enable_search_code": True,
                    }
                },
            )
        assert mock.call_args.kwargs["search_code_enabled"] is True
        names = [t["function"]["name"] for t in mock.call_args.kwargs["tools"]]
        assert names[-1] == "search_code"
        assert out["telemetry"]["retrieval_calls"] == 3


class TestAdaptResultV2:
    def test_includes_retrieval_fields(self) -> None:
        state = initial_state("bug")
        state["relevant_files"] = ["app/inventory.py"]
        state["telemetry"]["retrieval_calls"] = 1
        state["test_result"] = {"exit_code": 0}
        state["status"] = "success"
        out = adapt_result(state)
        assert out["relevant_files"] == ["app/inventory.py"]
        assert out["retrieval_calls"] == 1
