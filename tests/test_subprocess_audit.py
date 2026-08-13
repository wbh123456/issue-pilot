"""Audit: agent tools must not spawn host commands."""

from __future__ import annotations

import ast
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parent.parent

# Trusted orchestration only — never agent-visible tools.
_ALLOWED_SUBPROCESS_FILES = frozenset(
    {
        "sandbox/image.py",  # Docker CLI lifecycle / build
        "eval/repository.py",  # git reset --hard / git clean -fd
    }
)

_PRODUCTION_DIRS = ("agent", "eval", "harness", "sandbox")
_PRODUCTION_FILES = ("cli.py",)


def _rel(path: Path) -> str:
    return path.relative_to(HARNESS_ROOT).as_posix()


def _iter_production_py() -> list[Path]:
    files: list[Path] = []
    for folder in _PRODUCTION_DIRS:
        files.extend(
            p
            for p in (HARNESS_ROOT / folder).rglob("*.py")
            if "__pycache__" not in p.parts
        )
    for name in _PRODUCTION_FILES:
        path = HARNESS_ROOT / name
        if path.is_file():
            files.append(path)
    return files


def _imports_subprocess(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "subprocess" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "subprocess":
                return True
    return False


class TestSubprocessAudit:
    def test_only_trusted_orchestration_may_use_subprocess(self) -> None:
        offenders: list[str] = []
        for path in _iter_production_py():
            rel = _rel(path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            if not _imports_subprocess(tree):
                continue
            if rel not in _ALLOWED_SUBPROCESS_FILES:
                offenders.append(rel)
        assert offenders == [], (
            "subprocess is only allowed in Docker orchestration and trusted "
            f"evaluator reset: {_ALLOWED_SUBPROCESS_FILES}. Found in: {offenders}"
        )

    def test_agent_tools_have_no_subprocess(self) -> None:
        tools_dir = HARNESS_ROOT / "agent" / "tools"
        for path in tools_dir.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            assert not _imports_subprocess(tree), (
                f"{path.name} must not import subprocess; agent tools cannot "
                "spawn host commands"
            )
