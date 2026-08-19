"""Matrix orchestrator: split × harness × n, no live LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.matrix import parse_harness_list, parse_task_ids, run_matrix, tasks_for_split
from eval.runner import benchmark_spec_sha
from sandbox.image import DEFAULT_IMAGE

_FAKE_TASKS = [
    {
        "id": "issue-001",
        "split": "smoke",
        "base_commit": "abc123",
    },
    {
        "id": "issue-008",
        "split": "hard",
        "base_commit": "abc123",
    },
    {
        "id": "issue-009",
        "split": "hard",
        "base_commit": "abc123",
    },
    {
        "id": "issue-015",
        "split": "ablation",
        "base_commit": "abc123",
    },
]


def _solve_record(task_id: str, harness: str, *, success: bool = True) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "harness_version": harness,
        "success": success,
        "run_path": f"runs/{task_id}-{harness}.json",
        "paused": False,
    }


class TestParseHarnessList:
    def test_parses_and_dedupes(self) -> None:
        assert parse_harness_list("v0, v1,v0,v2") == ["v0", "v1", "v2"]

    def test_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="harness"):
            parse_harness_list("v9")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            parse_harness_list(" , ")


class TestTasksForSplit:
    def test_filters_hard(self) -> None:
        tasks = tasks_for_split("hard", _FAKE_TASKS)
        assert [t["id"] for t in tasks] == ["issue-008", "issue-009"]

    def test_filters_ablation(self) -> None:
        tasks = tasks_for_split("ablation", _FAKE_TASKS)
        assert [t["id"] for t in tasks] == ["issue-015"]

    def test_filters_task_ids_within_split(self) -> None:
        tasks = tasks_for_split(
            "hard", _FAKE_TASKS, task_ids="issue-009,issue-008"
        )
        assert [t["id"] for t in tasks] == ["issue-009", "issue-008"]

    def test_rejects_task_outside_split(self) -> None:
        with pytest.raises(ValueError, match="not in split"):
            tasks_for_split("hard", _FAKE_TASKS, task_ids=["issue-015"])

    def test_rejects_unknown_split(self) -> None:
        with pytest.raises(ValueError, match="split"):
            tasks_for_split("all", _FAKE_TASKS)


class TestParseTaskIds:
    def test_parses_and_dedupes(self) -> None:
        assert parse_task_ids("issue-008, issue-011,issue-008") == [
            "issue-008",
            "issue-011",
        ]

    def test_blank_is_none(self) -> None:
        assert parse_task_ids(None) is None

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            parse_task_ids(" , ")


class TestRunMatrix:
    def test_calls_solve_for_each_cell(self, tmp_path: Path) -> None:
        calls: list[tuple[str, str]] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            harness = kwargs["harness_version"]
            calls.append((task_id, harness))
            assert kwargs["model"] == "fake-model"
            assert kwargs["max_steps"] == 7
            assert kwargs["embedder_name"] == "hashing"
            assert kwargs["query_mode"] == "issue"
            assert kwargs["progress"] is None
            assert "require_approval" not in kwargs
            assert "interactive_recovery" not in kwargs
            return _solve_record(task_id, harness)

        manifest = run_matrix(
            split="hard",
            harnesses="v0,v1",
            n=1,
            model="fake-model",
            max_steps=7,
            runs_dir=tmp_path,
            solve=fake_solve,
            dataset=_FAKE_TASKS,
            progress=None,
        )
        assert calls == [
            ("issue-008", "v0"),
            ("issue-008", "v1"),
            ("issue-009", "v0"),
            ("issue-009", "v1"),
        ]
        assert len(manifest["cells"]) == 4
        assert all(cell["success"] is True for cell in manifest["cells"])
        assert manifest["settings"]["n"] == 1
        assert manifest["settings"]["benchmark_spec_sha"] == benchmark_spec_sha()
        log = Path(manifest["log_path"]).read_text(encoding="utf-8")
        assert "START issue-008 v0" in log
        assert "END issue-008 v0 exit=0" in log
        saved = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
        assert saved["cells"][0]["run_path"] == "runs/issue-008-v0.json"
        assert manifest["settings"]["task_ids"] == ["issue-008", "issue-009"]

    def test_filters_task_ids(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(task_id)
            return _solve_record(task_id, kwargs["harness_version"])

        manifest = run_matrix(
            split="hard",
            harnesses=["v0"],
            n=1,
            model="fake-model",
            runs_dir=tmp_path,
            solve=fake_solve,
            dataset=_FAKE_TASKS,
            task_ids="issue-009",
        )
        assert calls == ["issue-009"]
        assert manifest["settings"]["task_ids"] == ["issue-009"]

    def test_repeats_n_times(self, tmp_path: Path) -> None:
        calls: list[str] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs["harness_version"])
            return _solve_record(task_id, kwargs["harness_version"])

        run_matrix(
            split="hard",
            harnesses=["v2"],
            n=2,
            model="fake-model",
            runs_dir=tmp_path,
            solve=fake_solve,
            dataset=_FAKE_TASKS,
        )
        assert calls == ["v2", "v2", "v2", "v2"]

    def test_skip_existing_uses_matching_spec(self, tmp_path: Path) -> None:
        spec = benchmark_spec_sha()
        existing = {
            "task_id": "issue-008",
            "harness_version": "v0",
            "success": True,
            "base_commit": "abc123",
            "model": "fake-model",
            "temperature": 0,
            "sandbox_image": DEFAULT_IMAGE,
            "benchmark_spec_sha": spec,
        }
        (tmp_path / "issue-008-v0-20260801T000000Z.json").write_text(
            json.dumps(existing), encoding="utf-8"
        )
        calls: list[tuple[str, str]] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((task_id, kwargs["harness_version"]))
            return _solve_record(task_id, kwargs["harness_version"])

        manifest = run_matrix(
            split="hard",
            harnesses="v0",
            n=1,
            skip_existing=True,
            model="fake-model",
            runs_dir=tmp_path,
            solve=fake_solve,
            dataset=_FAKE_TASKS,
        )
        assert ("issue-008", "v0") not in calls
        assert calls == [("issue-009", "v0")]
        skipped = [cell for cell in manifest["cells"] if cell.get("skipped")]
        assert len(skipped) == 1
        assert skipped[0]["task_id"] == "issue-008"
        log = Path(manifest["log_path"]).read_text(encoding="utf-8")
        assert "SKIP issue-008 v0" in log

    def test_skip_existing_ignores_runs_without_spec_sha(self, tmp_path: Path) -> None:
        stale = {
            "task_id": "issue-008",
            "harness_version": "v0",
            "success": True,
            "base_commit": "abc123",
            "model": "fake-model",
            "temperature": 0,
            "sandbox_image": DEFAULT_IMAGE,
        }
        (tmp_path / "issue-008-v0-old.json").write_text(
            json.dumps(stale), encoding="utf-8"
        )
        calls: list[tuple[str, str]] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append((task_id, kwargs["harness_version"]))
            return _solve_record(task_id, kwargs["harness_version"])

        run_matrix(
            split="hard",
            harnesses="v0",
            n=1,
            skip_existing=True,
            model="fake-model",
            runs_dir=tmp_path,
            solve=fake_solve,
            dataset=_FAKE_TASKS,
        )
        assert ("issue-008", "v0") in calls

    def test_skip_existing_fills_remaining_repeats(self, tmp_path: Path) -> None:
        spec = benchmark_spec_sha()
        existing = {
            "task_id": "issue-008",
            "harness_version": "v0",
            "success": True,
            "base_commit": "abc123",
            "model": "fake-model",
            "temperature": 0,
            "sandbox_image": DEFAULT_IMAGE,
            "benchmark_spec_sha": spec,
        }
        (tmp_path / "issue-008-v0-20260801T000000Z.json").write_text(
            json.dumps(existing), encoding="utf-8"
        )
        calls: list[str] = []

        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            calls.append(task_id)
            return _solve_record(task_id, kwargs["harness_version"])

        manifest = run_matrix(
            split="hard",
            harnesses="v0",
            n=2,
            skip_existing=True,
            model="fake-model",
            runs_dir=tmp_path,
            solve=fake_solve,
            dataset=_FAKE_TASKS,
        )
        assert calls.count("issue-008") == 1
        assert calls.count("issue-009") == 2
        assert not any(cell.get("skipped") and cell["task_id"] == "issue-008" for cell in manifest["cells"])

    def test_records_solve_exception(self, tmp_path: Path) -> None:
        def fake_solve(task_id: str, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

        manifest = run_matrix(
            split="smoke",
            harnesses=["v0"],
            n=1,
            model="fake-model",
            runs_dir=tmp_path,
            solve=fake_solve,
            dataset=_FAKE_TASKS,
        )
        cell = manifest["cells"][0]
        assert cell["success"] is None
        assert "RuntimeError: boom" in cell["error"]
        log = Path(manifest["log_path"]).read_text(encoding="utf-8")
        assert "END issue-001 v0 exit=1" in log

    def test_rejects_n_less_than_one(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="n must be"):
            run_matrix(
                split="hard",
                n=0,
                runs_dir=tmp_path,
                solve=lambda *a, **k: _solve_record("x", "v0"),
                dataset=_FAKE_TASKS,
            )


class TestBenchmarkSpecSha:
    def test_changes_when_gold_changes(self, tmp_path: Path) -> None:
        dataset = tmp_path / "dataset.json"
        gold = tmp_path / "gold"
        gold.mkdir()
        dataset.write_bytes(b"[]")
        (gold / "test_issue_001.py").write_text("assert True\n", encoding="utf-8")
        first = benchmark_spec_sha(dataset_path=dataset, gold_dir=gold)
        (gold / "test_issue_001.py").write_text("assert False\n", encoding="utf-8")
        second = benchmark_spec_sha(dataset_path=dataset, gold_dir=gold)
        assert len(first) == 64
        assert first != second
