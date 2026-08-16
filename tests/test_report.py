"""Provenance-aware ablation report: cohort grouping, not mixed averages."""

from __future__ import annotations

import json
from pathlib import Path

from eval.report import (
    build_report,
    is_retrieve_record,
    is_solve_record,
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
        assert by_h["v0"]["resolve_rate"] == 0.5
        assert by_h["v0"]["tokens"] == 20.0
        assert by_h["v1"]["n"] == 1
        assert by_h["v1"]["resolve_rate"] == 1.0
        assert by_h["v0"]["recovery_rate"] == 0.0
        assert by_h["v0"]["human_retries"] == 0.0

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
