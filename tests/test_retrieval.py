"""Unit tests for the retrieval library. HashingEmbedder only — no model download."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.metrics import recall_at_k, unique_paths
from harness.context import (
    MAX_CHUNK_CHARS,
    MAX_PLANNER_CONTEXT_CHARS,
    RETRIEVE_K,
    RRF_K,
)
from retrieval.chunker import chunk_file, chunk_repo
from retrieval.dense import FastEmbedEmbedder, HashingEmbedder
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.indexer import build_index

INVENTORY_SRC = '''\
"""Warehouse stock and bin allocation."""

WAREHOUSE_BINS = {"widget": {"aisle": "A1", "slots": [0, 1, 2]}}


def get_stock(sku: str) -> int:
    return 3


def allocate_bin(items: list[dict]) -> str:
    """Pick a warehouse aisle and decrement stock."""
    qty = int(items[0]["qty"])
    slots = WAREHOUSE_BINS["widget"]["slots"]
    _ = slots[qty - 1]
    return "A1"


class Warehouse:
    def pick_slot(self, sku: str) -> int:
        return 0
'''

ORDERS_SRC = '''\
"""Create customer orders for widgets."""


def create_order(items: list[dict]) -> dict:
    return {"id": 1, "items": items}
'''

VALIDATORS_SRC = '''\
"""Email and field validators (distractor)."""


def deco(fn):
    return fn


@deco
def validate_email(value: str) -> bool:
    return "@" in value
'''

GOLD_SRC = "def definitely_secret_gold_symbol():\n    return 1\n"
BAK_SRC = "def leftover_backup_symbol():\n    return 1\n"
OUTSIDE_SRC = "def not_under_app():\n    return 1\n"


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _write(tmp_path, "app/inventory.py", INVENTORY_SRC)
    _write(tmp_path, "app/orders.py", ORDERS_SRC)
    _write(tmp_path, "app/validators.py", VALIDATORS_SRC)
    _write(tmp_path, "app/__init__.py", "")
    _write(tmp_path, "tests/_gold/secret.py", GOLD_SRC)
    _write(tmp_path, "app/_gold/hidden.py", GOLD_SRC)
    _write(tmp_path, "_app_bak/old.py", BAK_SRC)
    _write(tmp_path, "lib/other.py", OUTSIDE_SRC)
    _write(tmp_path, "README.md", "# toy\n")
    return tmp_path


def _by_symbol(repo: Path, symbol: str):
    matches = [c for c in chunk_repo(repo) if c.symbol == symbol]
    assert matches, f"missing chunk {symbol!r}"
    return matches[0]


class TestContextBudgets:
    def test_defaults(self) -> None:
        assert RETRIEVE_K == 5
        assert RRF_K == 60
        assert MAX_CHUNK_CHARS == 4_000
        assert MAX_PLANNER_CONTEXT_CHARS == 8_000


class TestChunker:
    def test_function_line_range(self, repo: Path) -> None:
        alloc = _by_symbol(repo, "allocate_bin")
        assert alloc.type == "function"
        assert alloc.path == "app/inventory.py"
        src = (repo / alloc.path).read_text(encoding="utf-8").splitlines()
        assert src[alloc.start_line - 1].startswith("def allocate_bin")
        body = "\n".join(src[alloc.start_line - 1 : alloc.end_line])
        assert "return \"A1\"" in body
        assert "class Warehouse" not in body

    def test_class_and_method(self, repo: Path) -> None:
        cls = _by_symbol(repo, "Warehouse")
        method = _by_symbol(repo, "Warehouse.pick_slot")
        assert cls.type == "class"
        assert method.type == "method"
        assert "def pick_slot" in cls.text
        src = (repo / method.path).read_text(encoding="utf-8").splitlines()
        assert "def pick_slot" in src[method.start_line - 1]

    def test_decorator_included_in_span(self, repo: Path) -> None:
        tagged = _by_symbol(repo, "validate_email")
        src = (repo / tagged.path).read_text(encoding="utf-8").splitlines()
        assert src[tagged.start_line - 1].strip() == "@deco"
        assert "@deco" in tagged.text
        assert "def validate_email" in tagged.text

    def test_gold_and_backup_not_indexed(self, repo: Path) -> None:
        chunks = chunk_repo(repo)
        symbols = {c.symbol for c in chunks}
        paths = {c.path for c in chunks}
        assert "definitely_secret_gold_symbol" not in symbols
        assert "leftover_backup_symbol" not in symbols
        assert "not_under_app" not in symbols
        assert all("_gold" not in p.split("/") for p in paths)
        assert all("_app_bak" not in p.split("/") for p in paths)
        assert "lib/other.py" not in paths
        assert "app/inventory.py" in paths
        assert "app/orders.py" in paths

    def test_long_chunk_is_clipped(self, tmp_path: Path) -> None:
        body = "    x = 1\n" * (MAX_CHUNK_CHARS // 2)
        _write(tmp_path, "app/huge.py", f"def huge():\n{body}")
        huge = _by_symbol(tmp_path, "huge")
        assert "truncated" in huge.text
        assert "MAX_CHUNK_CHARS" in huge.text
        assert huge.text.startswith("def huge():")


class TestPathJail:
    def test_relative_escape_rejected(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        (repo / "app").mkdir(parents=True)
        (repo / "app" / "ok.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        (tmp_path / "outside_secret.py").write_text(
            "def leaked():\n    return 1\n", encoding="utf-8"
        )
        with pytest.raises(PermissionError):
            chunk_file(repo, "../outside_secret.py")

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        (repo / "app").mkdir(parents=True)
        (repo / "app" / "ok.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        outsider = tmp_path / "secret.py"
        outsider.write_text("def leaked():\n    return 1\n", encoding="utf-8")
        with pytest.raises(PermissionError):
            chunk_file(repo, str(outsider))

    def test_symlink_outside_not_indexed(self, repo: Path, tmp_path: Path) -> None:
        outsider = tmp_path / "leaked.py"
        outsider.write_text("def leaked_symlink_symbol():\n    return 1\n", encoding="utf-8")
        link = repo / "app" / "leak.py"
        try:
            link.symlink_to(outsider)
        except OSError:
            pytest.skip("symlinks not available")
        symbols = {c.symbol for c in chunk_repo(repo)}
        assert "leaked_symlink_symbol" not in symbols


class TestBM25:
    def test_ranks_inventory_for_allocate_bin(self, repo: Path) -> None:
        index = build_index(repo)
        hits = index.search_bm25("allocate_bin warehouse slots", k=5)
        assert hits
        assert hits[0].path == "app/inventory.py"
        paths = unique_paths(c.path for c in hits)
        assert paths[0] == "app/inventory.py"


class TestDense:
    def test_hashing_embedder_is_default(self, repo: Path) -> None:
        index = build_index(repo)
        assert isinstance(index.embedder, HashingEmbedder)

    def test_fastembed_is_lazy(self) -> None:
        embedder = FastEmbedEmbedder()
        assert embedder._model is None

    def test_ranks_inventory_for_allocate_bin(self, repo: Path) -> None:
        index = build_index(repo, embedder=HashingEmbedder())
        hits = index.search_dense("allocate_bin warehouse slots", k=5)
        assert hits
        paths = unique_paths(c.path for c in hits)
        assert "app/inventory.py" in paths[:3]


class TestRRF:
    def test_prefers_doc_high_in_both_lists(self) -> None:
        fused = reciprocal_rank_fusion(
            [
                ["x", "y", "z"],
                ["x", "z", "y"],
            ],
            k=60,
        )
        assert fused[0][0] == "x"
        ids = [doc_id for doc_id, _score in fused]
        assert set(ids) == {"x", "y", "z"}

    def test_includes_docs_from_one_list_only(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b"], ["c"]], k=60)
        ids = [doc_id for doc_id, _score in fused]
        assert ids[0] == "a"
        assert set(ids) == {"a", "b", "c"}

    def test_hybrid_surfaces_expected_files(self, repo: Path) -> None:
        index = build_index(repo, embedder=HashingEmbedder())
        hits = index.search_hybrid("allocate_bin widgets create_order", k=5)
        paths = unique_paths(c.path for c in hits)
        assert "app/inventory.py" in paths
        expected = ["app/inventory.py", "app/orders.py"]
        assert recall_at_k(paths, expected, k=5) >= 0.5


class TestRecallAtK:
    def test_full_and_partial(self) -> None:
        retrieved = ["app/inventory.py", "app/validators.py", "app/orders.py"]
        expected = ["app/inventory.py", "app/orders.py"]
        assert recall_at_k(retrieved, expected, k=5) == 1.0
        assert recall_at_k(retrieved, expected, k=2) == 0.5
        assert recall_at_k(["app/inventory.py"], expected, k=5) == 0.5

    def test_collapses_duplicate_chunk_paths(self) -> None:
        retrieved = [
            "app/inventory.py",
            "app/inventory.py",
            "app/orders.py",
        ]
        expected = ["app/inventory.py", "app/orders.py"]
        assert recall_at_k(retrieved, expected, k=2) == 1.0

    def test_empty_expected_is_one(self) -> None:
        assert recall_at_k(["app/inventory.py"], [], k=5) == 1.0

    def test_empty_retrieved_is_zero(self) -> None:
        assert recall_at_k([], ["app/inventory.py"], k=5) == 0.0

    def test_rejects_non_positive_k(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k(["a"], ["a"], k=0)

    def test_on_fixture_index(self, repo: Path) -> None:
        index = build_index(repo, embedder=HashingEmbedder())
        hits = index.search_bm25("allocate_bin create_order widgets", k=5)
        score = recall_at_k(
            [c.path for c in hits],
            ["app/inventory.py", "app/orders.py"],
            k=5,
        )
        assert score == 1.0
