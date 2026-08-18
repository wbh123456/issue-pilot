"""Node-visit trajectory: append-only events, checkpoint stages, no chat dump."""

from __future__ import annotations

from unittest.mock import patch

from agent.nodes._runtime import append_trace, traced
from agent.nodes.execute import execute_plan
from agent.state import (
    CHECKPOINT_STAGE_LABELS,
    initial_state,
    reached_checkpoint_stages,
)

_EVENT_KEYS = {
    "node",
    "status",
    "detail",
    "retry_count",
    "tokens_delta",
    "timestamp",
}


class TestAppendTrace:
    def test_retries_append_never_overwrite(self) -> None:
        state = initial_state("bug")
        first = append_trace(state, node="plan", status="planned", detail="h1")
        assert [event["node"] for event in first] == ["plan"]
        assert set(first[0]) == _EVENT_KEYS
        assert first[0]["retry_count"] == 0
        assert first[0]["timestamp"].endswith("Z")

        retrying = {**state, "workflow_trace": first, "retry_count": 1}
        second = append_trace(retrying, node="plan", status="planned", detail="h2")
        assert [event["node"] for event in second] == ["plan", "plan"]
        assert [event["detail"] for event in second] == ["h1", "h2"]
        assert [event["retry_count"] for event in second] == [0, 1]

    def test_clips_detail_and_computes_token_delta(self) -> None:
        state = initial_state("bug")
        state["telemetry"] = {**state["telemetry"], "tokens": 10}
        out = traced(
            state,
            {
                "status": "analyzed",
                "telemetry": {**state["telemetry"], "tokens": 25},
            },
            node="analyze",
            detail="x" * 250,
        )
        event = out["workflow_trace"][-1]
        assert event["tokens_delta"] == 15
        assert len(event["detail"]) == 200
        assert event["detail"].endswith("...")


class TestCheckpointStages:
    def test_first_seen_node_boundaries(self) -> None:
        trace = [
            {"node": "analyze"},
            {"node": "plan"},
            {"node": "execute"},
            {"node": "verify"},
            {"node": "diagnose"},
            {"node": "plan"},
            {"node": "execute"},
            {"node": "verify"},
            {"node": "evaluate"},
            {"node": "await_approval"},
            {"node": "mark_success"},
        ]
        assert reached_checkpoint_stages(trace) == [
            CHECKPOINT_STAGE_LABELS["analyze"],
            CHECKPOINT_STAGE_LABELS["plan"],
            CHECKPOINT_STAGE_LABELS["execute"],
            CHECKPOINT_STAGE_LABELS["verify"],
            CHECKPOINT_STAGE_LABELS["await_approval"],
        ]

    def test_empty_trace(self) -> None:
        assert reached_checkpoint_stages([]) == []
        assert reached_checkpoint_stages(None) == []


class TestExecuteOmitsMessages:
    def test_raw_chat_stays_out_of_checkpointed_telemetry(self) -> None:
        state = initial_state("bug")
        executor = {
            "final_answer": "fixed",
            "termination": "completed",
            "steps": 1,
            "tool_call_count": 0,
            "file_reads": 0,
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "tokens": 4,
            "latency": 0.1,
            "trajectory": [{"tool": "read_file"}],
            "messages": [{"role": "user", "content": "huge chat dump"}],
        }
        with patch("agent.nodes.execute.run_agent", return_value=executor):
            out = execute_plan(
                state,
                {
                    "configurable": {
                        "client": object(),
                        "model": "m",
                        "repo_path": "/tmp/repo",
                        "test_command": "pytest -q",
                    }
                },
            )
        assert "messages" not in out["telemetry"]
        assert "huge chat dump" not in str(out["telemetry"])
        assert out["telemetry"]["trajectory"] == [{"tool": "read_file"}]
        assert out["workflow_trace"][-1]["node"] == "execute"
        assert out["workflow_trace"][-1]["detail"] == "steps=1"
