"""Same-process feedback node: skip, decline, one retry, refuse a second."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

from agent.graph import route_after_feedback
from agent.nodes.feedback import collect_feedback
from agent.nodes.plan import structured_plan
from agent.state import initial_state

HUMAN_FEEDBACK = "Catch ExpiredSignatureError and return 401"

VALID_PLAN = {
    "problem": "Expired JWT returns 500",
    "hypothesis": "decode_token misses ExpiredSignatureError",
    "files_to_inspect": ["app/auth.py"],
    "steps": ["inspect auth.py", "fix exception handling", "run tests"],
}

VALID_DIAGNOSIS = {
    "root_cause": "decode_token misses ExpiredSignatureError",
    "failure_category": "WRONG_HYPOTHESIS",
    "new_hypothesis": "Expired JWT must map to 401",
    "next_actions": ["catch ExpiredSignatureError", "return 401"],
}


class ScriptedProvider:
    def __init__(self, replies: list[str | None]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("unexpected feedback prompt")
        return self.replies.pop(0)


def _config(**extra: Any) -> dict:
    configurable = {"repo_path": "/tmp/repo", "model": "fake-model"}
    configurable.update(extra)
    return {"configurable": configurable}


def _exhausted_state() -> dict:
    state = initial_state("Expired JWT returns 500")
    state["diagnosis"] = "Need a tighter 401 mapping."
    state["retry_count"] = 2
    state["test_result"] = {"deterministic_pass": False}
    return state


class TestCollectFeedback:
    def test_no_provider_skips(self) -> None:
        out = collect_feedback(_exhausted_state(), _config())
        assert out["status"] == "feedback_skipped"
        assert "human_feedback" not in out
        assert route_after_feedback(out) == "mark_needs_human"

    def test_blank_and_none_decline(self) -> None:
        for reply in ("", "   ", None):
            provider = ScriptedProvider([reply])
            out = collect_feedback(
                _exhausted_state(),
                _config(feedback_provider=provider),
            )
            assert out["status"] == "feedback_declined"
            assert len(provider.prompts) == 1
            assert route_after_feedback(out) == "mark_needs_human"

    def test_accepts_one_retry_and_clips(self) -> None:
        provider = ScriptedProvider(["  " + HUMAN_FEEDBACK + "  "])
        out = collect_feedback(
            _exhausted_state(),
            _config(feedback_provider=provider),
        )
        assert out["status"] == "feedback_retry"
        assert out["human_feedback"] == HUMAN_FEEDBACK
        assert out["human_retry_count"] == 1
        assert "feedback_provider" not in out
        assert route_after_feedback(out) == "plan"
        assert "Expired JWT" in provider.prompts[0]

        long_text = "x" * 900
        out = collect_feedback(
            _exhausted_state(),
            _config(feedback_provider=ScriptedProvider([long_text])),
        )
        assert out["human_feedback"] == "x" * 800

    def test_refuses_second_attempt_without_calling_provider(self) -> None:
        provider = ScriptedProvider(["should not run"])
        state = _exhausted_state()
        state["human_retry_count"] = 1
        state["human_feedback"] = HUMAN_FEEDBACK
        out = collect_feedback(state, _config(feedback_provider=provider))
        assert out["status"] == "feedback_refused"
        assert provider.prompts == []
        assert route_after_feedback(out) == "mark_needs_human"


class TestPlanUsesFeedback:
    def test_replan_prompt_includes_human_feedback(self) -> None:
        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        class _Response:
            choices = [MagicMock(message=MagicMock(content=json.dumps(VALID_PLAN)))]
            usage = _Usage()

        client = MagicMock()
        client.chat.completions.create.return_value = _Response()
        state = initial_state("bug")
        state["plan"] = VALID_PLAN
        state["retry_count"] = 2
        state["structured_diagnosis"] = VALID_DIAGNOSIS
        state["human_feedback"] = HUMAN_FEEDBACK
        state["human_retry_count"] = 1
        with (
            patch("agent.nodes.plan.list_files", return_value="app/auth.py"),
            patch(
                "agent.nodes.plan.resolve_in_repo",
                side_effect=lambda repo, path: MagicMock(exists=lambda: True),
            ),
        ):
            structured_plan(
                state,
                {
                    "configurable": {
                        "client": client,
                        "model": "fake-model",
                        "repo_path": "/tmp/repo",
                    }
                },
            )
        user = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "Human feedback" in user
        assert HUMAN_FEEDBACK in user
