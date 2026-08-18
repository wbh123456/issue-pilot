"""Shared helpers for V1 LangGraph nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState, Telemetry
from harness.progress import ProgressReporter, get_reporter as resolve_reporter

_TRACE_DETAIL_LIMIT = 200


def configurable(config: RunnableConfig) -> dict[str, Any]:
    return dict(config.get("configurable") or {})


def require_config(config: RunnableConfig, *keys: str) -> dict[str, Any]:
    cfg = configurable(config)
    missing = [key for key in keys if key not in cfg or cfg[key] is None]
    if missing:
        raise RuntimeError(
            "Missing runtime config for node: " + ", ".join(missing)
        )
    return cfg


def get_reporter(config: RunnableConfig) -> ProgressReporter:
    return resolve_reporter(configurable(config).get("progress"))


def _merge_stage_tokens(
    current: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {
        stage: dict(usage)
        for stage, usage in dict(current.get("stage_tokens") or {}).items()
    }
    for stage, usage in incoming.items():
        existing = dict(merged.get(stage) or {})
        for key, value in dict(usage or {}).items():
            existing[key] = int(existing.get(key, 0) or 0) + int(value or 0)
        merged[stage] = existing
    return merged


def merge_telemetry(state: AgentState, **deltas: Any) -> Telemetry:
    current: dict[str, Any] = dict(state.get("telemetry") or {})
    for key, value in deltas.items():
        if value is None:
            continue
        if key == "stage_tokens":
            current["stage_tokens"] = _merge_stage_tokens(current, value)
        elif key in {
            "tool_call_count",
            "file_reads",
            "prompt_tokens",
            "completion_tokens",
            "tokens",
            "llm_calls",
            "steps",
            "latency",
            "retrieval_calls",
        }:
            current[key] = type(value)(current.get(key, 0) or 0) + value
        elif key == "trajectory":
            current[key] = list(current.get(key) or []) + list(value or [])
        else:
            current[key] = value
    current["tokens"] = int(current.get("prompt_tokens", 0) or 0) + int(
        current.get("completion_tokens", 0) or 0
    )
    return current  # type: ignore[return-value]


def _clip_detail(detail: str) -> str:
    text = " ".join(str(detail or "").split())
    if len(text) <= _TRACE_DETAIL_LIMIT:
        return text
    return text[: _TRACE_DETAIL_LIMIT - 3].rstrip() + "..."


def _token_count(telemetry: object) -> int:
    if not isinstance(telemetry, dict):
        return 0
    return int(telemetry.get("tokens") or 0)


def append_trace(
    state: AgentState,
    *,
    node: str,
    status: str,
    detail: str = "",
    retry_count: int | None = None,
    tokens_delta: int = 0,
) -> list[dict[str, Any]]:
    """Return the existing visit log plus one new event (retries append)."""
    event: dict[str, Any] = {
        "node": node,
        "status": status,
        "detail": _clip_detail(detail),
        "retry_count": int(
            retry_count if retry_count is not None else (state.get("retry_count") or 0)
        ),
        "tokens_delta": int(tokens_delta or 0),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return list(state.get("workflow_trace") or []) + [event]


def traced(
    state: AgentState,
    update: dict[str, Any],
    *,
    node: str,
    detail: str = "",
    status: str | None = None,
    tokens_delta: int | None = None,
) -> dict[str, Any]:
    """Copy ``update`` and attach a full ``workflow_trace`` (no list reducer)."""
    out = dict(update)
    event_status = (
        status
        if status is not None
        else str(out.get("status") or state.get("status") or "")
    )
    retry_count = (
        int(out["retry_count"])
        if "retry_count" in out
        else int(state.get("retry_count") or 0)
    )
    if tokens_delta is None:
        before = _token_count(state.get("telemetry"))
        after = _token_count(out.get("telemetry")) if "telemetry" in out else before
        tokens_delta = max(0, after - before)
    out["workflow_trace"] = append_trace(
        state,
        node=node,
        status=event_status,
        detail=detail,
        retry_count=retry_count,
        tokens_delta=tokens_delta,
    )
    return out


def usage_deltas(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    return {
        "llm_calls": 1,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }


def stage_usage(stage: str, response: Any) -> dict[str, Any]:
    """Usage deltas plus a mergeable ``stage_tokens`` fragment for one LLM stage."""
    deltas = usage_deltas(response)
    return {
        **deltas,
        "stage_tokens": {
            stage: {
                "prompt_tokens": deltas["prompt_tokens"],
                "completion_tokens": deltas["completion_tokens"],
                "llm_calls": 1,
            }
        },
    }
