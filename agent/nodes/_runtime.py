"""Shared helpers for V1 LangGraph nodes."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState, Telemetry


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
