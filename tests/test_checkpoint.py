"""Durable SqliteSaver checkpoints and shared runtime config."""

from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END, START, StateGraph

from agent.graph import (
    build_graph,
    build_runtime_config,
    get_graph,
    get_v2_graph,
    run_workflow,
)
from harness.checkpoint import (
    DEFAULT_CHECKPOINT_PATH,
    checkpoint_path,
    open_checkpointer,
)
from harness.limits import GRAPH_RECURSION_LIMIT, MAX_AGENT_STEPS
from tests.test_graph import (
    FakeClient,
    PASS_PATCH,
    VALID_EVALUATION,
    VALID_PLAN,
    _Response,
    _fake_list_files,
    _layer1_run_command,
)


class _CounterState(TypedDict):
    value: int


def _linear_graph(checkpointer):
    graph = StateGraph(_CounterState)
    graph.add_node("a", lambda state: {"value": state["value"] + 1})
    graph.add_node("b", lambda state: {"value": state["value"] + 10})
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    return graph.compile(checkpointer=checkpointer)


def _completed_nodes(compiled, config: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for snapshot in reversed(list(compiled.get_state_history(config))):
        for task in snapshot.tasks:
            if task.name != "__start__" and task.result is not None:
                names.append(task.name)
    return names


def _pass_patches():
    return (
        patch("agent.nodes.plan.list_files", side_effect=_fake_list_files),
        patch(
            "agent.nodes.plan.resolve_in_repo",
            side_effect=lambda repo, path: MagicMock(exists=lambda: True),
        ),
        patch(
            "agent.nodes.execute.run_agent",
            return_value={
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
            },
        ),
        patch(
            "agent.nodes.verify.run_command",
            side_effect=_layer1_run_command(),
        ),
        patch(
            "agent.nodes.verify.inspect_worktree",
            return_value=PASS_PATCH,
        ),
        patch("agent.graph.diagnose_failure"),
    )


class TestCheckpointPath:
    def test_default_is_runs_sqlite(self) -> None:
        assert checkpoint_path() == DEFAULT_CHECKPOINT_PATH
        assert DEFAULT_CHECKPOINT_PATH.name == "checkpoints.sqlite"
        assert DEFAULT_CHECKPOINT_PATH.parent.name == "runs"


class TestRuntimeConfig:
    def test_includes_recursion_limit_and_original_keys(self) -> None:
        client = object()
        cfg = build_runtime_config(
            client=client,
            repo_path="/tmp/repo",
            test_command="pytest -q",
        )
        assert cfg["recursion_limit"] == GRAPH_RECURSION_LIMIT
        configurable = cfg["configurable"]
        assert configurable["client"] is client
        assert configurable["repo_path"] == "/tmp/repo"
        assert configurable["test_command"] == "pytest -q"
        assert configurable["lint_command"] == "ruff check app"
        assert configurable["max_steps"] == MAX_AGENT_STEPS
        assert configurable["enable_search_code"] is False
        assert configurable["embedder_name"] == "hashing"
        assert configurable["query_mode"] == "issue"
        assert configurable["require_approval"] is False
        assert "thread_id" not in configurable

    def test_thread_id_only_when_set(self) -> None:
        cfg = build_runtime_config(
            client=object(),
            repo_path="/tmp/repo",
            test_command="pytest -q",
            thread_id="issue-001-v1-test",
            recursion_limit=50,
        )
        assert cfg["recursion_limit"] == 50
        assert cfg["configurable"]["thread_id"] == "issue-001-v1-test"


class TestDurableCheckpointer:
    def test_checkpoint_after_each_node(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        config = {"configurable": {"thread_id": "linear-1"}}
        with open_checkpointer(db) as saver:
            compiled = _linear_graph(saver)
            compiled.invoke({"value": 0}, config)
            assert _completed_nodes(compiled, config) == ["a", "b"]
            assert compiled.get_state(config).values["value"] == 11

    def test_state_reloads_from_fresh_saver(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        config = {"configurable": {"thread_id": "linear-2"}}
        with open_checkpointer(db) as saver:
            _linear_graph(saver).invoke({"value": 0}, config)

        with open_checkpointer(db) as saver:
            compiled = _linear_graph(saver)
            snapshot = compiled.get_state(config)
            assert snapshot.values["value"] == 11
            assert snapshot.next == ()
            assert _completed_nodes(compiled, config) == ["a", "b"]

    def test_v1_graph_reloads_final_state(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        thread_id = "issue-001-v1-ckpt"
        client = FakeClient(
            [
                _Response("Problem: expired JWT. Hypothesis: missing catch."),
                _Response(json.dumps(VALID_PLAN)),
                _Response(json.dumps(VALID_EVALUATION)),
            ]
        )
        with open_checkpointer(db) as saver, ExitStack() as stack:
            for patcher in _pass_patches():
                stack.enter_context(patcher)
            result = run_workflow(
                client=client,
                issue="Expired JWT returns 500",
                repo_path="/tmp/repo",
                test_command="pytest -q",
                model="fake-model",
                max_steps=5,
                graph=build_graph(checkpointer=saver),
                thread_id=thread_id,
            )
        assert result["status"] == "success"
        assert result["workflow_passed"] is True

        config = build_runtime_config(
            client=object(),
            repo_path="/tmp/repo",
            test_command="pytest -q",
            thread_id=thread_id,
        )
        with open_checkpointer(db) as saver:
            compiled = build_graph(checkpointer=saver)
            snapshot = compiled.get_state(config)
            assert snapshot.values["status"] == "success"
            assert snapshot.next == ()
            assert _completed_nodes(compiled, config) == [
                "analyze",
                "plan",
                "execute",
                "verify",
                "evaluate",
                "await_approval",
                "mark_success",
            ]

    def test_checkpointed_graph_requires_thread_id(self, tmp_path) -> None:
        db = tmp_path / "ck.sqlite"
        with open_checkpointer(db) as saver:
            with pytest.raises(ValueError, match="thread_id is required"):
                run_workflow(
                    client=FakeClient([]),
                    issue="x",
                    repo_path="/tmp/repo",
                    test_command="pytest -q",
                    graph=build_graph(checkpointer=saver),
                )

    def test_singleton_stays_uncheckpointed(self, tmp_path) -> None:
        v1 = get_graph()
        v2 = get_v2_graph()
        with open_checkpointer(tmp_path / "ck.sqlite") as saver:
            compiled = build_graph(checkpointer=saver)
            assert compiled is not v1
            assert compiled.checkpointer is saver
        assert get_graph() is v1
        assert get_v2_graph() is v2
        assert get_graph().checkpointer is None
        assert get_v2_graph().checkpointer is None
