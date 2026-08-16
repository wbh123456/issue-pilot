"""Aggregate saved run JSON into provenance-aware ablation tables.

Solve records are grouped by a cohort of ``base_commit``, ``model``,
``temperature``, and ``sandbox_image`` so mixed-benchmark runs are not
averaged together. Retrieval eval artifacts (``*-retrieve-*.json``) are
reported separately.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.runner import HARNESS_ROOT, RUNS_DIR

COHORT_KEYS = ("base_commit", "model", "temperature", "sandbox_image")


def load_run_files(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    root = Path(runs_dir or RUNS_DIR)
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        data.setdefault("run_path", str(path))
        records.append(data)
    return records


def is_solve_record(record: dict[str, Any]) -> bool:
    return "harness_version" in record and "success" in record and "modes" not in record


def is_retrieve_record(record: dict[str, Any]) -> bool:
    modes = record.get("modes")
    if not isinstance(modes, dict) or not modes:
        return False
    first = next(iter(modes.values()), None)
    return isinstance(first, dict) and "recall_at_k" in first


def cohort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(key) for key in COHORT_KEYS)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _recovery_success_flag(record: dict[str, Any]) -> float:
    """1.0 if this solve recovered after a retry. Missing field is derived."""
    if "recovery_success" in record:
        return 1.0 if record.get("recovery_success") else 0.0
    retry = int(record.get("retry_count") or 0)
    return 1.0 if retry > 0 and record.get("workflow_passed") is True else 0.0


def _fmt(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def summarize_solve_cohort(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per harness_version inside a single provenance cohort."""
    by_harness: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not is_solve_record(record):
            continue
        by_harness[str(record.get("harness_version") or "v0")].append(record)

    rows: list[dict[str, Any]] = []
    for harness in ("v0", "v1", "v2"):
        group = by_harness.get(harness) or []
        if not group:
            continue
        successes = [1.0 if r.get("success") else 0.0 for r in group]
        tokens = [float(r["tokens"]) for r in group if r.get("tokens") is not None]
        reads = [float(r["file_reads"]) for r in group if r.get("file_reads") is not None]
        tools = [
            float(r["tool_call_count"])
            for r in group
            if r.get("tool_call_count") is not None
        ]
        retries = [
            float(r["retry_count"])
            for r in group
            if r.get("retry_count") is not None
        ]
        recovery = [_recovery_success_flag(r) for r in group]
        human = [float(r.get("human_retry_count") or 0) for r in group]
        latency = [float(r["latency"]) for r in group if r.get("latency") is not None]
        rows.append(
            {
                "harness_version": harness,
                "n": len(group),
                "resolve_rate": _mean(successes),
                "tokens": _mean(tokens),
                "file_reads": _mean(reads),
                "tool_calls": _mean(tools),
                "retries": _mean(retries) if retries else 0.0,
                "recovery_rate": _mean(recovery) if recovery else 0.0,
                "human_retries": _mean(human) if human else 0.0,
                "latency_s": _mean(latency),
            }
        )
    return rows


def summarize_retrieval(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not is_retrieve_record(record):
            continue
        modes = record.get("modes") or {}
        row: dict[str, Any] = {
            "task_id": record.get("task_id"),
            "embedder": record.get("embedder"),
            "query_mode": record.get("query_mode"),
        }
        for mode, payload in modes.items():
            row[mode] = (payload or {}).get("recall_at_k")
        rows.append(row)
    return rows


def build_report(
    *,
    runs_dir: Path | None = None,
    split: str | None = None,
    base_commit: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    records = load_run_files(runs_dir)
    solves = [r for r in records if is_solve_record(r)]
    retrieves = [r for r in records if is_retrieve_record(r)]

    if split:
        solves = [r for r in solves if r.get("split") == split]
        retrieves = [r for r in retrieves if r.get("split") == split]
    if base_commit:
        solves = [r for r in solves if r.get("base_commit") == base_commit]
        retrieves = [
            r for r in retrieves if r.get("base_commit") in {None, base_commit}
        ]
    if model:
        solves = [r for r in solves if r.get("model") == model]

    cohorts: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in solves:
        cohorts[cohort_key(record)].append(record)

    cohort_rows = []
    for key, group in sorted(cohorts.items(), key=lambda item: str(item[0])):
        meta = dict(zip(COHORT_KEYS, key))
        cohort_rows.append(
            {
                **meta,
                "n": len(group),
                "harnesses": summarize_solve_cohort(group),
            }
        )

    return {
        "runs_dir": str(Path(runs_dir or RUNS_DIR)),
        "filters": {
            "split": split,
            "base_commit": base_commit,
            "model": model,
        },
        "solve_cohorts": cohort_rows,
        "retrieval": summarize_retrieval(retrieves),
        "harness_git_sha": next(
            (r.get("harness_git_sha") for r in solves if r.get("harness_git_sha")),
            None,
        ),
        "harness_root": str(HARNESS_ROOT),
    }
