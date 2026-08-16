"""Same-process human feedback after the automatic retry budget is exhausted."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from harness.limits import MAX_HUMAN_RETRY
from harness.progress import preview

from ._runtime import configurable, get_reporter

FeedbackProvider = Callable[[str], str | None]

_FEEDBACK_LIMIT = 800


def stdin_feedback_provider(prompt: str) -> str | None:
    """Read one line from stdin. Blank or EOF means the operator declined."""
    print(prompt)
    try:
        line = input("Recovery feedback (blank to escalate): ")
    except EOFError:
        return None
    text = line.strip()
    return text or None


def _feedback_prompt(state: AgentState) -> str:
    test_result = state.get("test_result") or {}
    diagnosis = (state.get("diagnosis") or "").strip() or "(none)"
    return (
        "Automatic retry budget exhausted. Optional recovery feedback.\n"
        f"Issue:\n{state.get('issue') or ''}\n\n"
        f"Diagnosis:\n{diagnosis[:_FEEDBACK_LIMIT]}\n\n"
        f"retry_count={int(state.get('retry_count') or 0)} "
        f"deterministic_pass={test_result.get('deterministic_pass')}\n"
        "Enter guidance for one more replan, or leave blank to escalate.\n"
    )


def collect_feedback(state: AgentState, config: RunnableConfig) -> dict:
    """Ask the runtime provider at most once; never store the callback in state."""
    reporter = get_reporter(config)
    if int(state.get("human_retry_count") or 0) >= MAX_HUMAN_RETRY:
        reporter.stage("feedback", "refused")
        return {"status": "feedback_refused"}

    provider = configurable(config).get("feedback_provider")
    if provider is None:
        reporter.stage("feedback", "skipped")
        return {"status": "feedback_skipped"}

    raw = provider(_feedback_prompt(state))
    text = (raw or "").strip()[:_FEEDBACK_LIMIT]
    if not text:
        reporter.stage("feedback", "declined")
        return {"status": "feedback_declined"}

    reporter.stage("feedback", preview(text))
    return {
        "human_feedback": text,
        "human_retry_count": int(state.get("human_retry_count") or 0) + 1,
        "status": "feedback_retry",
    }
