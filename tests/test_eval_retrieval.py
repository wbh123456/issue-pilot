"""Retrieval eval CLI helpers. HashingEmbedder only — no model download."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eval.retrieval import (
    ALL_MODES,
    evaluate_repo,
    grep_query_tokens,
    rank_files_by_grep,
    run_retrieval_eval,
    save_retrieve_run,
)
from retrieval.dense import HashingEmbedder
from retrieval.lexical import BM25Index

INVENTORY_SRC = '''\
"""Warehouse stock and bin allocation."""

def allocate_bin(items: list[dict]) -> str:
    return "A1"
'''

ORDERS_SRC = '''\
"""Create customer orders for widgets."""

def create_order(items: list[dict]) -> dict:
    return {"id": 1, "items": items}
'''

VALIDATORS_SRC = "widget\n" * 12 + "\ndef validate_email(value: str) -> bool:\n    return True\n"


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _write(tmp_path, "app/inventory.py", INVENTORY_SRC)
    _write(tmp_path, "app/orders.py", ORDERS_SRC)
    _write(tmp_path, "app/validators.py", VALIDATORS_SRC)
    return tmp_path


def _task(repo: Path) -> dict[str, Any]:
    return {
        "id": "issue-009",
        "split": "hard",
        "issue": "Ordering 50 widgets when 3 are in stock crashes with 500",
        "expected_files": ["app/inventory.py", "app/orders.py"],
        "base_commit": "deadbeef",
        "repo_path": str(repo),
    }


class TestGrepBaseline:
    def test_drops_stopwords_and_short_tokens(self) -> None:
        tokens = grep_query_tokens(
            "Ordering 50 widgets when 3 are in stock crashes with 500"
        )
        assert tokens == ["Ordering", "widgets", "stock", "crashes", "500"]

    def test_ranks_by_hit_count_not_bm25(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        original = BM25Index.search

        def wrapped(self: BM25Index, query: str, k: int = 5) -> list:
            called.append(query)
            return original(self, query, k=k)

        monkeypatch.setattr(BM25Index, "search", wrapped)
        ranked = rank_files_by_grep(repo, "please find the widget")
        assert called == []
        assert ranked[0] == "app/validators.py"
        assert "app/orders.py" in ranked

    def test_is_not_the_same_as_bm25(self, repo: Path) -> None:
        record = evaluate_repo(
            {
                "id": "issue-toy",
                "issue": "please find the widget allocate_bin",
                "expected_files": ["app/inventory.py"],
            },
            repo,
            embedder=HashingEmbedder(),
            k=5,
        )
        grep_files = record["modes"]["grep"]["retrieved_files"]
        bm25_files = record["modes"]["bm25"]["retrieved_files"]
        assert grep_files[0] == "app/validators.py"
        assert bm25_files[0] == "app/inventory.py"


class TestEvaluateRepo:
    def test_four_modes_and_recall(self, repo: Path) -> None:
        record = evaluate_repo(
            _task(repo),
            repo,
            embedder=HashingEmbedder(),
            k=5,
        )
        assert set(record["modes"]) == set(ALL_MODES)
        assert record["embedder"] == "HashingEmbedder"
        assert record["chunk_count"] >= 1
        expected = ["app/inventory.py", "app/orders.py"]
        assert record["modes"]["bm25"]["recall_at_k"] == 1.0
        for mode in ALL_MODES:
            score = record["modes"][mode]["recall_at_k"]
            assert 0.0 <= score <= 1.0
            files = record["modes"][mode]["retrieved_files"]
            assert len(files) <= 5
        assert "app/orders.py" in record["modes"]["bm25"]["retrieved_files"]
        assert expected == record["expected_files"]


class TestRunRetrievalEval:
    def test_split_saves_retrieve_json(
        self,
        repo: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runs = tmp_path / "runs"
        monkeypatch.setattr("eval.retrieval.RUNS_DIR", runs)
        monkeypatch.setattr("eval.retrieval._select_tasks", lambda **kwargs: [_task(repo)])
        monkeypatch.setattr("eval.retrieval._resolve_repo_path", lambda task: repo)
        reset_calls: list[tuple[Path, str]] = []
        monkeypatch.setattr(
            "eval.retrieval.reset_repo",
            lambda path, commit: reset_calls.append((path, commit)),
        )

        result = run_retrieval_eval(
            split="hard",
            embedder_name="hashing",
            reset=True,
            save=True,
            k=5,
        )

        assert reset_calls == [(repo, "deadbeef")]
        assert result["embedder"] == "HashingEmbedder"
        assert result["split"] == "hard"
        assert len(result["tasks"]) == 1
        row = result["tasks"][0]
        assert row["task_id"] == "issue-009"
        run_path = Path(row["run_path"])
        assert run_path.parent == runs
        assert "issue-009-retrieve-" in run_path.name
        assert run_path.suffix == ".json"
        assert set(result["mean_recall_at_k"]) == set(ALL_MODES)

    def test_requires_task_or_split(self) -> None:
        with pytest.raises(ValueError, match="task_id or --split"):
            run_retrieval_eval(embedder_name="hashing", reset=False, save=False)

    def test_save_retrieve_run_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eval.retrieval.RUNS_DIR", tmp_path)
        path = save_retrieve_run({"task_id": "issue-009", "modes": {}})
        assert path.name.startswith("issue-009-retrieve-")
        assert path.suffix == ".json"


class TestCLIRetrieve:
    def test_retrieve_does_not_import_langgraph(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        sys.modules.pop("langgraph", None)
        import cli

        seen: dict[str, Any] = {}

        def fake_eval(**kwargs: Any) -> dict[str, Any]:
            seen.update(kwargs)
            return {
                "split": "hard",
                "k": 5,
                "embedder": "HashingEmbedder",
                "tasks": [
                    {
                        "task_id": "issue-009",
                        "issue": "Ordering 50 widgets",
                        "expected_files": ["app/inventory.py"],
                        "modes": {
                            mode: {"recall_at_k": 1.0, "retrieved_files": []}
                            for mode in ALL_MODES
                        },
                    }
                ],
                "mean_recall_at_k": {mode: 1.0 for mode in ALL_MODES},
            }

        monkeypatch.setattr(cli, "run_retrieval_eval", fake_eval)
        assert cli.main(["retrieve", "--split", "hard", "--embedder", "hashing"]) == 0
        assert seen["split"] == "hard"
        assert seen["embedder_name"] == "hashing"
