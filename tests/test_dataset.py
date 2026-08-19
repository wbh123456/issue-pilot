"""Dataset schema, leak checks, and hidden-gold staging."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from eval.runner import (
    GOLD_DIR,
    HARNESS_ROOT,
    cleanup_gold_staging,
    gold_staging_dir,
    load_dataset,
    run_gold_test,
    stage_gold_file,
)
from sandbox.runner import CommandResult

BENCHMARK_ROOT = (HARNESS_ROOT / ".." / "issue-pilot-benchmark").resolve()
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_LEAK_RE = re.compile(r"GOLD:|issue-\d{3}", re.IGNORECASE)
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HISTORICAL_LEAKS = frozenset({"coworker", "docs"})


def _idents(text: str) -> set[str]:
    return {match.group(0).lower() for match in _IDENT_RE.finditer(text)}


def _def_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name.lower())
    return names


def _dict_keys(path: Path, target_names: frozenset[str]) -> set[str]:
    if not path.is_file():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()

    def _take(name: str, value: ast.AST) -> None:
        if name not in target_names or not isinstance(value, ast.Dict):
            return
        for key in value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value.lower())

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    _take(target.id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                _take(node.target.id, node.value)
    return keys


def _sku_and_coupon_literals() -> set[str]:
    """SKU and coupon codes — grep bait if they appear in issue text."""
    names = set()
    names.update(
        _dict_keys(
            BENCHMARK_ROOT / "app" / "inventory.py",
            frozenset({"_DEFAULT_STOCK", "WAREHOUSE_BINS"}),
        )
    )
    names.update(
        _dict_keys(
            BENCHMARK_ROOT / "app" / "pricing.py",
            frozenset({"COUPONS"}),
        )
    )
    return names


def _forbidden_hard_idents(task: dict) -> set[str]:
    forbidden = set(_HISTORICAL_LEAKS)
    forbidden.update(_sku_and_coupon_literals())
    for rel in task.get("expected_files") or []:
        forbidden.add(Path(rel).stem.lower())
        path = BENCHMARK_ROOT / rel
        if path.is_file():
            forbidden.update(_def_names(path))
    return forbidden


class TestDataset:
    def test_tasks_have_hidden_gold_and_split(self) -> None:
        tasks = load_dataset()
        ids = [t["id"] for t in tasks]
        assert ids == [f"issue-{i:03d}" for i in range(1, 15)]
        splits = {t["split"] for t in tasks}
        assert splits == {"smoke", "hard"}
        for task in tasks:
            gold = GOLD_DIR / task["gold_file"]
            assert gold.is_file(), task["id"]
            text = gold.read_text(encoding="utf-8")
            assert f"def {task['gold_test']}(" in text
            assert _SHA_RE.match(task["base_commit"]), task["id"]
            assert task["test_command"].startswith("pytest tests/")
            assert task["lint_command"].startswith("ruff ")
            assert "::_" not in task["test_command"]
            assert task["gold_file"] not in task["test_command"]
            assert task["gold_file"] not in task["lint_command"]

    def test_visible_tests_do_not_name_issues_or_gold(self) -> None:
        tests_dir = BENCHMARK_ROOT / "tests"
        assert tests_dir.is_dir()
        for path in tests_dir.glob("test_*.py"):
            text = path.read_text(encoding="utf-8")
            assert _LEAK_RE.search(text) is None, path.name

    def test_readme_has_no_issue_map(self) -> None:
        readme = (BENCHMARK_ROOT / "README.md").read_text(encoding="utf-8")
        assert "issue-001" not in readme
        assert "GOLD" not in readme

    def test_hard_prompts_do_not_spell_the_fix(self) -> None:
        tasks = [task for task in load_dataset() if task.get("split") == "hard"]
        assert tasks, "expected a hard split"
        for task in tasks:
            issue_idents = _idents(task["issue"])
            leaked = sorted(issue_idents & _forbidden_hard_idents(task))
            assert leaked == [], f"{task['id']} issue text leaks {leaked}"

        eleven = next(task for task in tasks if task["id"] == "issue-011")
        forbidden = _forbidden_hard_idents(eleven)
        assert "save10" in forbidden
        assert "widget" in forbidden
        assert "pricing" in forbidden
        assert "apply_coupon" in forbidden
        assert "price" not in forbidden

        inventory = (BENCHMARK_ROOT / "app" / "inventory.py").read_text(
            encoding="utf-8"
        )
        assert "IndexError" not in inventory
        assert "HTTP 500" not in inventory
        assert "Checkout is supposed to take stock through" not in inventory

    def test_indexed_app_has_at_least_150_chunks(self) -> None:
        from retrieval.chunker import chunk_repo

        assert len(chunk_repo(BENCHMARK_ROOT)) >= 150


class TestGoldStaging:
    def test_stage_copies_then_cleanup_removes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "bench"
        (repo / "tests").mkdir(parents=True)
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        (gold_dir / "test_issue_001.py").write_text(
            "def test_expired_token_returns_401():\n    assert True\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("eval.runner.GOLD_DIR", gold_dir)
        task = {
            "id": "issue-001",
            "gold_file": "test_issue_001.py",
            "gold_test": "test_expired_token_returns_401",
        }
        staged = stage_gold_file(repo, task)
        assert staged.is_file()
        assert staged.parent == gold_staging_dir(repo)
        cleanup_gold_staging(repo)
        assert not gold_staging_dir(repo).exists()

    def test_run_gold_test_targets_hidden_file_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "bench"
        (repo / "tests").mkdir(parents=True)
        (repo / "tests" / "test_auth_expired.py").write_text(
            "def test_stale_session_does_not_crash():\n    assert False\n",
            encoding="utf-8",
        )
        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        (gold_dir / "test_issue_001.py").write_text(
            "def test_expired_token_returns_401():\n    assert True\n",
            encoding="utf-8",
        )
        monkeypatch.setattr("eval.runner.GOLD_DIR", gold_dir)

        class FakeSandbox:
            def __init__(self) -> None:
                self.commands: list[str] = []

            def run(self, command: str | list[str]) -> CommandResult:
                text = command if isinstance(command, str) else " ".join(command)
                self.commands.append(text)
                argv = text.split() if isinstance(command, str) else list(command)
                return CommandResult(command=argv, exit_code=0, stdout="1 passed", stderr="")

        sandbox = FakeSandbox()
        task = {
            "id": "issue-001",
            "gold_file": "test_issue_001.py",
            "gold_test": "test_expired_token_returns_401",
            "test_command": "pytest tests/test_auth_expired.py -q",
        }
        result = run_gold_test(repo, task, sandbox=sandbox)

        assert len(sandbox.commands) == 1
        cmd = sandbox.commands[0]
        assert "tests/_gold/test_issue_001.py::test_expired_token_returns_401" in cmd
        assert "test_auth_expired" not in cmd
        assert result["passed"] is True
        assert not gold_staging_dir(repo).exists()
