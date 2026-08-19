"""Aggregate saved run JSON into provenance-aware ablation tables.

Solve records are grouped by a cohort of ``base_commit``, ``model``,
``temperature``, ``sandbox_image``, and ``benchmark_spec_sha`` so mixed
benchmark or gold revisions are not averaged together. Retrieval eval
artifacts (``*-retrieve-*.json``) are reported separately.

Harness rows average **cells** ``(task_id, harness_version)``, not raw runs,
so a task with extra seeds cannot dominate the mean.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.metrics import normalize_path, unique_paths
from eval.runner import HARNESS_ROOT, RUNS_DIR

COHORT_KEYS = (
    "base_commit",
    "model",
    "temperature",
    "sandbox_image",
    "benchmark_spec_sha",
)

_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")
_METRIC_FIELDS = (
    "resolve_rate",
    "tokens",
    "file_reads",
    "tool_calls",
    "retries",
    "recovery_rate",
    "human_retries",
    "latency_s",
    "localization_precision",
    "layer1_gate_rate",
    "search_code_calls",
    "first_expected_read_step",
)

_DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/", re.MULTILINE)
_EXIT_ZERO_RE = re.compile(r"exit_code\s*=\s*0")


def load_run_files(runs_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load top-level ``runs/*.json`` only (not ``archived/``, ``sessions/``, or ``matrix-*.json``)."""
    root = Path(runs_dir or RUNS_DIR)
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("matrix-"):
            continue
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


def patched_files_from_diff(diff: str) -> list[str]:
    """Paths from a unified ``git diff`` header, preserving order."""
    return unique_paths(_DIFF_FILE_RE.findall(diff or ""))


def localization_precision(record: dict[str, Any]) -> float | None:
    """Changed files ∩ expected_files / changed files. None if the patch is empty."""
    expected = {
        normalize_path(path)
        for path in (record.get("expected_files") or [])
        if path
    }
    patched = patched_files_from_diff(str(record.get("patch_diff") or ""))
    if not patched:
        return None
    hits = sum(1 for path in patched if path in expected)
    return hits / len(patched)


def search_code_calls(record: dict[str, Any]) -> float:
    return float(
        sum(
            1
            for event in (record.get("trajectory") or [])
            if (event or {}).get("tool") == "search_code"
        )
    )


def first_expected_read_step(record: dict[str, Any]) -> float | None:
    """Earliest trajectory step that read an expected file. None if never."""
    expected = {
        normalize_path(path)
        for path in (record.get("expected_files") or [])
        if path
    }
    if not expected:
        return None
    best: float | None = None
    for event in record.get("trajectory") or []:
        if (event or {}).get("tool") != "read_file":
            continue
        raw = ((event or {}).get("arguments") or {}).get("path")
        path = normalize_path(str(raw or ""))
        if path not in expected:
            continue
        step = (event or {}).get("step")
        if step is None:
            continue
        value = float(step)
        if best is None or value < best:
            best = value
    return best


def layer1_gate_rate(record: dict[str, Any]) -> float | None:
    """1.0 if the delivered patch still fails visible tests.

    v1/v2 go through verify, so this is ``not pytest_passed``. v0 has no
    verify node; the last ``run_tests`` tool result is used instead.
    """
    harness = str(record.get("harness_version") or "v0")
    if harness in {"v1", "v2"}:
        verification = record.get("verification") or {}
        if "pytest_passed" in verification:
            return 0.0 if verification.get("pytest_passed") else 1.0
        if "workflow_passed" in record:
            return 0.0 if record.get("workflow_passed") else 1.0
        return None
    last = None
    for event in record.get("trajectory") or []:
        if (event or {}).get("tool") == "run_tests":
            last = event
    if last is None:
        return 1.0
    result = str(last.get("result") or "")
    return 0.0 if _EXIT_ZERO_RE.search(result) else 1.0


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


def _record_sort_key(record: dict[str, Any]) -> tuple[str, float, str]:
    path = str(record.get("run_path") or "")
    name = Path(path).name
    match = _STAMP_RE.search(name)
    stamp = match.group(1) if match else ""
    mtime = 0.0
    try:
        mtime = Path(path).stat().st_mtime
    except OSError:
        pass
    return (stamp, mtime, name)


def select_latest_per_cell(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one solve per (cohort, task_id, harness_version): the newest file."""
    chosen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            *cohort_key(record),
            str(record.get("task_id") or ""),
            str(record.get("harness_version") or "v0"),
        )
        prev = chosen.get(key)
        if prev is None or _record_sort_key(record) > _record_sort_key(prev):
            chosen[key] = record
    return list(chosen.values())


def _summarize_runs(group: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [1.0 if r.get("success") else 0.0 for r in group]
    tokens = [float(r["tokens"]) for r in group if r.get("tokens") is not None]
    reads = [float(r["file_reads"]) for r in group if r.get("file_reads") is not None]
    tools = [
        float(r["tool_call_count"])
        for r in group
        if r.get("tool_call_count") is not None
    ]
    retries = [
        float(r["retry_count"]) for r in group if r.get("retry_count") is not None
    ]
    recovery = [_recovery_success_flag(r) for r in group]
    human = [float(r.get("human_retry_count") or 0) for r in group]
    latency = [float(r["latency"]) for r in group if r.get("latency") is not None]
    loc = [v for r in group if (v := localization_precision(r)) is not None]
    gate = [v for r in group if (v := layer1_gate_rate(r)) is not None]
    search = [search_code_calls(r) for r in group]
    first_read = [v for r in group if (v := first_expected_read_step(r)) is not None]
    return {
        "n": len(group),
        "resolve_rate": _mean(successes),
        "tokens": _mean(tokens),
        "file_reads": _mean(reads),
        "tool_calls": _mean(tools),
        "retries": _mean(retries) if retries else 0.0,
        "recovery_rate": _mean(recovery) if recovery else 0.0,
        "human_retries": _mean(human) if human else 0.0,
        "latency_s": _mean(latency),
        "localization_precision": _mean(loc),
        "layer1_gate_rate": _mean(gate),
        "search_code_calls": _mean(search) if search else 0.0,
        "first_expected_read_step": _mean(first_read),
    }


def summarize_solve_cohort(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per harness_version inside a single provenance cohort.

    Each ``(task_id, harness_version)`` cell is averaged first (seed mean),
    then those cell means are averaged to the harness row so extra runs on
    one task cannot dominate.
    """
    by_harness: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if not is_solve_record(record):
            continue
        harness = str(record.get("harness_version") or "v0")
        task_id = str(record.get("task_id") or "")
        by_harness[harness][task_id].append(record)

    rows: list[dict[str, Any]] = []
    for harness in ("v0", "v1", "v2"):
        cells_map = by_harness.get(harness) or {}
        if not cells_map:
            continue
        cells: list[dict[str, Any]] = []
        for task_id in sorted(cells_map):
            cell = _summarize_runs(cells_map[task_id])
            cell["task_id"] = task_id
            cells.append(cell)
        averaged: dict[str, Any] = {
            "harness_version": harness,
            "n": sum(int(cell["n"]) for cell in cells),
            "n_cells": len(cells),
            "cells": cells,
        }
        for field in _METRIC_FIELDS:
            averaged[field] = _mean(
                [float(cell[field]) for cell in cells if cell.get(field) is not None]
            )
        if averaged["retries"] is None:
            averaged["retries"] = 0.0
        if averaged["recovery_rate"] is None:
            averaged["recovery_rate"] = 0.0
        if averaged["human_retries"] is None:
            averaged["human_retries"] = 0.0
        if averaged["search_code_calls"] is None:
            averaged["search_code_calls"] = 0.0
        rows.append(averaged)
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
    latest_per_cell: bool = False,
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
    if latest_per_cell:
        solves = select_latest_per_cell(solves)

    cohorts: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in solves:
        cohorts[cohort_key(record)].append(record)

    cohort_rows = []
    for key, group in sorted(cohorts.items(), key=lambda item: str(item[0])):
        meta = dict(zip(COHORT_KEYS, key))
        harnesses = summarize_solve_cohort(group)
        cohort_rows.append(
            {
                **meta,
                "n": len(group),
                "n_cells": sum(int(row.get("n_cells") or 0) for row in harnesses),
                "harnesses": harnesses,
            }
        )

    return {
        "runs_dir": str(Path(runs_dir or RUNS_DIR)),
        "filters": {
            "split": split,
            "base_commit": base_commit,
            "model": model,
            "latest_per_cell": latest_per_cell,
        },
        "solve_cohorts": cohort_rows,
        "retrieval": summarize_retrieval(retrieves),
        "harness_git_sha": next(
            (r.get("harness_git_sha") for r in solves if r.get("harness_git_sha")),
            None,
        ),
        "harness_root": str(HARNESS_ROOT),
    }
