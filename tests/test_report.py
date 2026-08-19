"""Provenance-aware ablation report: cohort grouping, not mixed averages."""

from __future__ import annotations

import json
from pathlib import Path

from eval.report import (
    build_report,
    first_expected_read_step,
    is_retrieve_record,
    is_solve_record,
    layer1_gate_rate,
    load_run_files,
    localization_precision,
    search_code_calls,
    summarize_solve_cohort,
)


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _solve(
    *,
    task_id: str,
    harness: str,
    success: bool,
    base_commit: str = "aaa",
    model: str = "deepseek-v4-flash",
    temperature: int = 0,
    sandbox_image: str = "issue-pilot-sandbox:py312",
    split: str = "hard",
    tokens: int = 100,
    retry_count: int = 0,
    workflow_passed: bool | None = None,
    recovery_success: bool | None = None,
    human_retry_count: int | None = None,
) -> dict:
    record = {
        "task_id": task_id,
        "harness_version": harness,
        "success": success,
        "base_commit": base_commit,
        "model": model,
        "temperature": temperature,
        "sandbox_image": sandbox_image,
        "split": split,
        "tokens": tokens,
        "file_reads": 2,
        "tool_call_count": 4,
        "retry_count": retry_count,
        "latency": 1.0,
        "harness_git_sha": "harness-sha",
    }
    if workflow_passed is not None:
        record["workflow_passed"] = workflow_passed
    if recovery_success is not None:
        record["recovery_success"] = recovery_success
    if human_retry_count is not None:
        record["human_retry_count"] = human_retry_count
    return record


def _retrieve(*, task_id: str, split: str = "hard", base_commit: str = "aaa") -> dict:
    return {
        "task_id": task_id,
        "split": split,
        "base_commit": base_commit,
        "embedder": "HashingEmbedder",
        "query_mode": "issue",
        "modes": {
            "grep": {"recall_at_k": 0.5},
            "bm25": {"recall_at_k": 0.75},
            "dense": {"recall_at_k": 1.0},
            "hybrid": {"recall_at_k": 1.0},
        },
    }


class TestRecordKinds:
    def test_solve_vs_retrieve(self) -> None:
        solve = _solve(task_id="issue-009", harness="v1", success=True)
        retrieve = _retrieve(task_id="issue-009")
        assert is_solve_record(solve)
        assert not is_retrieve_record(solve)
        assert is_retrieve_record(retrieve)
        assert not is_solve_record(retrieve)


class TestCohortSummary:
    def test_groups_by_harness(self) -> None:
        rows = summarize_solve_cohort(
            [
                _solve(task_id="issue-008", harness="v0", success=True, tokens=10),
                _solve(task_id="issue-009", harness="v0", success=False, tokens=30),
                _solve(task_id="issue-008", harness="v1", success=True, tokens=20),
            ]
        )
        by_h = {row["harness_version"]: row for row in rows}
        assert by_h["v0"]["n"] == 2
        assert by_h["v0"]["n_cells"] == 2
        assert by_h["v0"]["resolve_rate"] == 0.5
        assert by_h["v0"]["tokens"] == 20.0
        assert by_h["v1"]["n"] == 1
        assert by_h["v1"]["n_cells"] == 1
        assert by_h["v1"]["resolve_rate"] == 1.0
        assert by_h["v0"]["recovery_rate"] == 0.0
        assert by_h["v0"]["human_retries"] == 0.0
        assert {cell["task_id"] for cell in by_h["v0"]["cells"]} == {
            "issue-008",
            "issue-009",
        }

    def test_averages_cells_not_raw_runs(self) -> None:
        rows = summarize_solve_cohort(
            [
                _solve(task_id="issue-001", harness="v1", success=True, tokens=10),
                _solve(task_id="issue-001", harness="v1", success=True, tokens=90),
                _solve(task_id="issue-002", harness="v1", success=False, tokens=100),
            ]
        )
        row = rows[0]
        assert row["n"] == 3
        assert row["n_cells"] == 2
        assert row["resolve_rate"] == 0.5
        assert row["tokens"] == 75.0
        by_task = {cell["task_id"]: cell for cell in row["cells"]}
        assert by_task["issue-001"]["tokens"] == 50.0
        assert by_task["issue-001"]["n"] == 2
        assert by_task["issue-002"]["tokens"] == 100.0

    def test_recovery_rate_and_human_retries(self) -> None:
        rows = summarize_solve_cohort(
            [
                _solve(task_id="issue-008", harness="v1", success=True, retry_count=0),
                _solve(
                    task_id="issue-009",
                    harness="v1",
                    success=True,
                    retry_count=1,
                    workflow_passed=True,
                ),
                _solve(
                    task_id="issue-010",
                    harness="v1",
                    success=True,
                    retry_count=1,
                    recovery_success=True,
                    human_retry_count=1,
                ),
                _solve(
                    task_id="issue-011",
                    harness="v1",
                    success=False,
                    retry_count=2,
                    recovery_success=False,
                ),
            ]
        )
        row = rows[0]
        assert row["recovery_rate"] == 0.5
        assert row["human_retries"] == 0.25


class TestBuildReport:
    def test_does_not_average_across_base_commits(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "a-v0.json",
            _solve(task_id="issue-009", harness="v0", success=True, base_commit="old"),
        )
        _write(
            tmp_path,
            "b-v0.json",
            _solve(task_id="issue-009", harness="v0", success=False, base_commit="new"),
        )
        report = build_report(runs_dir=tmp_path)
        assert len(report["solve_cohorts"]) == 2
        commits = {c["base_commit"] for c in report["solve_cohorts"]}
        assert commits == {"old", "new"}
        for cohort in report["solve_cohorts"]:
            assert cohort["n"] == 1
            assert len(cohort["harnesses"]) == 1

    def test_filters_split_and_keeps_retrieve_separate(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "hard-v1.json",
            _solve(
                task_id="issue-009",
                harness="v1",
                success=True,
                split="hard",
                retry_count=1,
            ),
        )
        _write(
            tmp_path,
            "smoke-v1.json",
            _solve(
                task_id="issue-001",
                harness="v1",
                success=False,
                split="smoke",
            ),
        )
        _write(tmp_path, "issue-009-retrieve-x.json", _retrieve(task_id="issue-009"))
        report = build_report(runs_dir=tmp_path, split="hard")
        assert len(report["solve_cohorts"]) == 1
        assert report["solve_cohorts"][0]["n"] == 1
        assert report["solve_cohorts"][0]["harnesses"][0]["retries"] == 1.0
        assert report["solve_cohorts"][0]["harnesses"][0]["recovery_rate"] == 0.0
        assert report["solve_cohorts"][0]["harnesses"][0]["human_retries"] == 0.0
        assert len(report["retrieval"]) == 1
        assert report["retrieval"][0]["hybrid"] == 1.0
        assert report["retrieval"][0]["grep"] == 0.5
        assert report["harness_git_sha"] == "harness-sha"
        assert report["filters"]["split"] == "hard"

    def test_skips_corrupt_json(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
        _write(
            tmp_path,
            "ok.json",
            _solve(task_id="issue-009", harness="v2", success=True),
        )
        report = build_report(runs_dir=tmp_path)
        assert report["solve_cohorts"][0]["n"] == 1

    def test_ignores_session_sidecars_in_subdirectory(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "ok.json",
            _solve(task_id="issue-009", harness="v1", success=True),
        )
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        (sessions / "issue-009-v1-paused.json").write_text(
            json.dumps(
                _solve(task_id="issue-009", harness="v1", success=False),
            ),
            encoding="utf-8",
        )
        records = load_run_files(tmp_path)
        assert len(records) == 1
        assert records[0]["success"] is True
        report = build_report(runs_dir=tmp_path)
        assert report["solve_cohorts"][0]["n"] == 1
        assert report["solve_cohorts"][0]["harnesses"][0]["resolve_rate"] == 1.0

    def test_splits_cohorts_on_benchmark_spec_sha(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "old-spec.json",
            _solve(
                task_id="issue-009",
                harness="v1",
                success=True,
                tokens=10,
            )
            | {"benchmark_spec_sha": "aaa"},
        )
        _write(
            tmp_path,
            "new-spec.json",
            _solve(
                task_id="issue-009",
                harness="v1",
                success=False,
                tokens=90,
            )
            | {"benchmark_spec_sha": "bbb"},
        )
        report = build_report(runs_dir=tmp_path)
        assert len(report["solve_cohorts"]) == 2
        shas = {c["benchmark_spec_sha"] for c in report["solve_cohorts"]}
        assert shas == {"aaa", "bbb"}
        for cohort in report["solve_cohorts"]:
            assert cohort["n"] == 1
            assert cohort["n_cells"] == 1

    def test_latest_per_cell_drops_older_duplicate(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "issue-001-v1-20260101T000000Z.json",
            _solve(task_id="issue-001", harness="v1", success=False, tokens=10),
        )
        _write(
            tmp_path,
            "issue-001-v1-20260801T120000Z.json",
            _solve(task_id="issue-001", harness="v1", success=True, tokens=90),
        )
        mixed = build_report(runs_dir=tmp_path)
        assert mixed["solve_cohorts"][0]["n"] == 2
        assert mixed["solve_cohorts"][0]["harnesses"][0]["n_cells"] == 1
        assert mixed["solve_cohorts"][0]["harnesses"][0]["tokens"] == 50.0

        latest = build_report(runs_dir=tmp_path, latest_per_cell=True)
        row = latest["solve_cohorts"][0]["harnesses"][0]
        assert latest["filters"]["latest_per_cell"] is True
        assert latest["solve_cohorts"][0]["n"] == 1
        assert row["n"] == 1
        assert row["n_cells"] == 1
        assert row["tokens"] == 90.0
        assert row["resolve_rate"] == 1.0

    def test_skips_matrix_manifests(self, tmp_path: Path) -> None:
        _write(
            tmp_path,
            "ok.json",
            _solve(task_id="issue-009", harness="v1", success=True),
        )
        _write(
            tmp_path,
            "matrix-20260819T000000Z.json",
            {
                "stamp": "20260819T000000Z",
                "cells": [],
                "harness_version": "v0",
                "success": False,
            },
        )
        records = load_run_files(tmp_path)
        assert len(records) == 1
        assert records[0]["task_id"] == "issue-009"


class TestDerivedMetrics:
    def test_localization_precision_from_patch_diff(self) -> None:
        record = _solve(task_id="issue-008", harness="v1", success=True) | {
            "expected_files": ["app/orders.py"],
            "patch_diff": (
                "diff --git a/app/orders.py b/app/orders.py\n+ok\n"
                "diff --git a/app/auth.py b/app/auth.py\n+lint\n"
            ),
        }
        assert localization_precision(record) == 0.5

    def test_localization_precision_skips_empty_diff(self) -> None:
        record = _solve(task_id="issue-008", harness="v0", success=False) | {
            "expected_files": ["app/orders.py"],
            "patch_diff": "",
        }
        assert localization_precision(record) is None

    def test_search_code_calls_and_first_expected_read(self) -> None:
        record = _solve(task_id="issue-015", harness="v2", success=True) | {
            "expected_files": ["app/tax.py"],
            "trajectory": [
                {
                    "step": 1,
                    "tool": "search_code",
                    "arguments": {"query": "surcharge"},
                },
                {
                    "step": 2,
                    "tool": "read_file",
                    "arguments": {"path": "app/orders.py"},
                },
                {
                    "step": 4,
                    "tool": "read_file",
                    "arguments": {"path": "app/tax.py"},
                },
                {
                    "step": 5,
                    "tool": "search_code",
                    "arguments": {"query": "Nexus"},
                },
            ],
        }
        assert search_code_calls(record) == 2.0
        assert first_expected_read_step(record) == 4.0

    def test_layer1_gate_rate_v1_uses_pytest_passed(self) -> None:
        passed = _solve(task_id="issue-015", harness="v1", success=True) | {
            "verification": {"pytest_passed": True},
        }
        failed = _solve(task_id="issue-015", harness="v1", success=False) | {
            "verification": {"pytest_passed": False},
        }
        assert layer1_gate_rate(passed) == 0.0
        assert layer1_gate_rate(failed) == 1.0

    def test_layer1_gate_rate_v0_uses_last_run_tests(self) -> None:
        delivered = _solve(task_id="issue-015", harness="v0", success=False) | {
            "trajectory": [
                {
                    "step": 2,
                    "tool": "run_tests",
                    "result": "exit_code=1\nFAILED",
                },
                {
                    "step": 5,
                    "tool": "run_tests",
                    "result": "exit_code=0\n1 passed",
                },
            ]
        }
        skipped = _solve(task_id="issue-015", harness="v0", success=False) | {
            "trajectory": [{"step": 1, "tool": "read_file", "arguments": {}}],
        }
        assert layer1_gate_rate(delivered) == 0.0
        assert layer1_gate_rate(skipped) == 1.0

    def test_cohort_averages_derived_metrics(self) -> None:
        rows = summarize_solve_cohort(
            [
                _solve(task_id="issue-015", harness="v2", success=True)
                | {
                    "expected_files": ["app/tax.py"],
                    "patch_diff": "diff --git a/app/tax.py b/app/tax.py\n+ok\n",
                    "verification": {"pytest_passed": True},
                    "trajectory": [
                        {
                            "step": 1,
                            "tool": "search_code",
                            "arguments": {"query": "levy"},
                        },
                        {
                            "step": 2,
                            "tool": "read_file",
                            "arguments": {"path": "app/tax.py"},
                        },
                    ],
                },
                _solve(task_id="issue-016", harness="v2", success=False)
                | {
                    "expected_files": ["app/notifications.py"],
                    "patch_diff": (
                        "diff --git a/app/notifications.py "
                        "b/app/notifications.py\n+ok\n"
                        "diff --git a/app/auth.py b/app/auth.py\n+lint\n"
                    ),
                    "verification": {"pytest_passed": False},
                    "trajectory": [
                        {
                            "step": 3,
                            "tool": "read_file",
                            "arguments": {"path": "app/orders.py"},
                        }
                    ],
                },
            ]
        )
        row = rows[0]
        assert row["localization_precision"] == 0.75
        assert row["layer1_gate_rate"] == 0.5
        assert row["search_code_calls"] == 0.5
        assert row["first_expected_read_step"] == 2.0
