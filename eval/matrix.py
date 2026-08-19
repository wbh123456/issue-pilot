"""Deterministic V0/V1/V2 matrix orchestrator. Does not call the LLM itself."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from agent.client import default_model
from eval.report import is_solve_record, load_run_files
from eval.runner import RUNS_DIR, benchmark_spec_sha, load_dataset, solve_task
from harness.limits import AGENT_TEMPERATURE, MAX_AGENT_STEPS
from retrieval.query import DEFAULT_QUERY_MODE
from sandbox.image import DEFAULT_IMAGE

SolveFn = Callable[..., dict[str, Any]]
VALID_HARNESSES = ("v0", "v1", "v2")
VALID_SPLITS = ("smoke", "hard")


def parse_harness_list(raw: str) -> list[str]:
    parts = [item.strip().lower() for item in (raw or "").split(",") if item.strip()]
    if not parts:
        raise ValueError("at least one harness is required")
    seen: list[str] = []
    for item in parts:
        if item not in VALID_HARNESSES:
            raise ValueError(
                f"harness must be one of {', '.join(VALID_HARNESSES)}, got {item!r}"
            )
        if item not in seen:
            seen.append(item)
    return seen


def tasks_for_split(split: str, dataset: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be 'smoke' or 'hard', got {split!r}")
    tasks = dataset if dataset is not None else load_dataset()
    selected = [task for task in tasks if task.get("split") == split]
    if not selected:
        raise ValueError(f"no tasks in split {split!r}")
    return selected


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_log_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _matching_runs(
    records: list[dict[str, Any]],
    *,
    task_id: str,
    harness: str,
    base_commit: str,
    model: str,
    spec_sha: str,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    for record in records:
        if not is_solve_record(record):
            continue
        if str(record.get("task_id")) != task_id:
            continue
        if str(record.get("harness_version") or "v0") != harness:
            continue
        if str(record.get("base_commit") or "") != base_commit:
            continue
        if str(record.get("model") or "") != model:
            continue
        if record.get("temperature") != AGENT_TEMPERATURE:
            continue
        if str(record.get("sandbox_image") or "") != DEFAULT_IMAGE:
            continue
        if record.get("benchmark_spec_sha") != spec_sha:
            continue
        matched.append(record)
    matched.sort(key=lambda record: str(record.get("run_path") or ""))
    return matched


def _append_log(handle: TextIO | None, line: str) -> None:
    if handle is None:
        return
    handle.write(line + "\n")
    handle.flush()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_matrix(
    *,
    split: str,
    harnesses: list[str] | str = "v0,v1,v2",
    n: int = 1,
    skip_existing: bool = False,
    model: str | None = None,
    max_steps: int = MAX_AGENT_STEPS,
    embedder_name: str = "hashing",
    query_mode: str = DEFAULT_QUERY_MODE,
    log_path: Path | str | None = None,
    runs_dir: Path | None = None,
    solve: SolveFn | None = None,
    progress=None,
    dataset: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run ``split`` × harnesses × n solves. Writes a log and a manifest.

    Does not pass ``require_approval`` or ``interactive_recovery``. ``solve`` is
    injectable so tests never call a live LLM.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    harness_list = (
        parse_harness_list(harnesses) if isinstance(harnesses, str) else list(harnesses)
    )
    if not harness_list:
        raise ValueError("at least one harness is required")
    for item in harness_list:
        if item not in VALID_HARNESSES:
            raise ValueError(
                f"harness must be one of {', '.join(VALID_HARNESSES)}, got {item!r}"
            )

    tasks = tasks_for_split(split, dataset)
    model_name = model or default_model()
    spec_sha = benchmark_spec_sha()
    stamp = _utc_stamp()
    root = Path(runs_dir or RUNS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"matrix-{stamp}.json"
    resolved_log = Path(log_path) if log_path else root / f"matrix-{stamp}.log"
    solved = solve or solve_task
    existing_records = load_run_files(root) if skip_existing else []

    base_commits = sorted({str(task.get("base_commit") or "") for task in tasks})
    settings = {
        "split": split,
        "harnesses": harness_list,
        "n": n,
        "model": model_name,
        "temperature": AGENT_TEMPERATURE,
        "max_steps": max_steps,
        "sandbox_image": DEFAULT_IMAGE,
        "embedder": embedder_name,
        "query_mode": query_mode,
        "skip_existing": skip_existing,
        "benchmark_spec_sha": spec_sha,
        "base_commits": base_commits,
    }
    manifest: dict[str, Any] = {
        "stamp": stamp,
        "log_path": str(resolved_log),
        "manifest_path": str(manifest_path),
        "settings": settings,
        "cells": [],
    }

    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    with resolved_log.open("w", encoding="utf-8") as log:
        _append_log(
            log,
            f"settings: model={model_name} temperature={AGENT_TEMPERATURE} "
            f"max_steps={max_steps}",
        )
        _append_log(log, f"base_commit={','.join(base_commits)}")
        _append_log(log, f"sandbox={DEFAULT_IMAGE}")
        _append_log(log, f"v2: embedder={embedder_name} query_mode={query_mode}")
        _append_log(log, f"n={n} per cell")
        _append_log(log, f"benchmark_spec_sha={spec_sha}")

        for task in tasks:
            task_id = str(task["id"])
            base_commit = str(task.get("base_commit") or "")
            for harness in harness_list:
                matching = _matching_runs(
                    existing_records,
                    task_id=task_id,
                    harness=harness,
                    base_commit=base_commit,
                    model=model_name,
                    spec_sha=spec_sha,
                )
                already = len(matching)
                if skip_existing and already >= n:
                    latest = matching[-1]
                    run_path = str(latest.get("run_path") or "")
                    _append_log(
                        log,
                        f"{_utc_log_time()} SKIP {task_id} {harness} existing={run_path}",
                    )
                    manifest["cells"].append(
                        {
                            "task_id": task_id,
                            "harness_version": harness,
                            "skipped": True,
                            "n_existing": already,
                            "run_path": run_path,
                            "success": latest.get("success"),
                            "error": None,
                        }
                    )
                    _write_manifest(manifest_path, manifest)
                    continue

                to_run = n - already if skip_existing else n
                for seed in range(to_run):
                    _append_log(log, f"{_utc_log_time()} START {task_id} {harness}")
                    cell: dict[str, Any] = {
                        "task_id": task_id,
                        "harness_version": harness,
                        "seed": already + seed if skip_existing else seed,
                        "skipped": False,
                        "run_path": None,
                        "success": None,
                        "error": None,
                    }
                    try:
                        record = solved(
                            task_id,
                            model=model_name,
                            max_steps=max_steps,
                            harness_version=harness,
                            embedder_name=embedder_name,
                            query_mode=query_mode,
                            progress=progress,
                        )
                    except Exception as exc:
                        cell["error"] = f"{type(exc).__name__}: {exc}"
                        _append_log(
                            log,
                            f"{_utc_log_time()} END {task_id} {harness} exit=1",
                        )
                    else:
                        cell["run_path"] = record.get("run_path")
                        cell["success"] = record.get("success")
                        cell["paused"] = bool(record.get("paused"))
                        exit_code = 0 if record.get("success") else 1
                        _append_log(
                            log,
                            f"{_utc_log_time()} END {task_id} {harness} exit={exit_code}",
                        )
                    manifest["cells"].append(cell)
                    _write_manifest(manifest_path, manifest)

    return manifest
