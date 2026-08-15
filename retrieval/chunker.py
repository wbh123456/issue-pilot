"""AST symbol chunker for Python sources under ``app/``."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent.tools._sandbox import (
    rel_to_repo,
    resolve_in_repo,
    resolve_repo_root,
    should_skip_dir,
)
from harness.context import MAX_CHUNK_CHARS, truncate_chars

ChunkKind = Literal["module", "class", "function", "method"]

INDEX_DIR = "app"

__all__ = [
    "INDEX_DIR",
    "Chunk",
    "ChunkKind",
    "chunk_file",
    "chunk_repo",
    "index_text",
    "iter_app_python_files",
]


@dataclass(frozen=True)
class Chunk:
    path: str
    symbol: str
    type: ChunkKind
    start_line: int
    end_line: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}:{self.symbol}"


def index_text(chunk: Chunk) -> str:
    """Text fed to BM25 / dense indexes (path + symbol + body)."""
    return f"{chunk.path}\n{chunk.symbol}\n{chunk.type}\n{chunk.text}"


def iter_app_python_files(repo_path: str | Path) -> list[Path]:
    """Path-jailed ``app/**/*.py``, skipping ``_gold`` / ``_app_bak`` / VCS dirs."""
    root = resolve_repo_root(repo_path)
    app_dir = root / INDEX_DIR
    if not app_dir.is_dir():
        return []

    files: list[Path] = []
    for path in sorted(app_dir.rglob("*.py")):
        if not path.is_file():
            continue
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            continue
        if any(should_skip_dir(part) for part in rel.parts):
            continue
        try:
            resolve_in_repo(root, rel.as_posix())
        except PermissionError:
            continue
        files.append(path)
    return files


def chunk_repo(repo_path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_app_python_files(repo_path):
        rel = rel_to_repo(repo_path, path)
        chunks.extend(chunk_file(repo_path, rel))
    return chunks


def chunk_file(repo_path: str | Path, rel_path: str) -> list[Chunk]:
    target = resolve_in_repo(repo_path, rel_path)
    if not target.is_file():
        return []
    try:
        source = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    rel = rel_to_repo(repo_path, target)
    return _chunk_source(rel, source)


def _chunk_source(rel_path: str, source: str) -> list[Chunk]:
    nlines = len(source.splitlines()) or 1
    module_symbol = Path(rel_path).stem
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            _make_chunk(rel_path, module_symbol, "module", 1, nlines, source)
        ]

    lines = source.splitlines(keepends=True)
    chunks: list[Chunk] = []
    leftover: list[ast.AST] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            chunks.extend(_chunk_class(rel_path, lines, node, parent=""))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                _function_chunk(rel_path, lines, node, kind="function", parent="")
            )
        else:
            leftover.append(node)

    if leftover:
        start, end, text = _join_nodes(lines, leftover)
        chunks.insert(
            0,
            _make_chunk(rel_path, module_symbol, "module", start, end, text),
        )
    return chunks


def _chunk_class(
    rel_path: str,
    lines: list[str],
    node: ast.ClassDef,
    *,
    parent: str,
) -> list[Chunk]:
    qname = f"{parent}.{node.name}" if parent else node.name
    start, end = _node_span(node)
    chunks = [
        _make_chunk(rel_path, qname, "class", start, end, _slice(lines, start, end))
    ]
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(
                _function_chunk(
                    rel_path, lines, item, kind="method", parent=qname
                )
            )
        elif isinstance(item, ast.ClassDef):
            chunks.extend(_chunk_class(rel_path, lines, item, parent=qname))
    return chunks


def _function_chunk(
    rel_path: str,
    lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    kind: Literal["function", "method"],
    parent: str,
) -> Chunk:
    name = f"{parent}.{node.name}" if parent else node.name
    start, end = _node_span(node)
    return _make_chunk(rel_path, name, kind, start, end, _slice(lines, start, end))


def _make_chunk(
    path: str,
    symbol: str,
    kind: ChunkKind,
    start_line: int,
    end_line: int,
    text: str,
) -> Chunk:
    return Chunk(
        path=path,
        symbol=symbol,
        type=kind,
        start_line=start_line,
        end_line=end_line,
        text=truncate_chars(text, MAX_CHUNK_CHARS, label="MAX_CHUNK_CHARS"),
    )


def _node_span(node: ast.AST) -> tuple[int, int]:
    start = int(getattr(node, "lineno", 1) or 1)
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        start = min(start, min(int(d.lineno) for d in decorators))
    end = int(getattr(node, "end_lineno", start) or start)
    return start, max(end, start)


def _join_nodes(lines: list[str], nodes: list[ast.AST]) -> tuple[int, int, str]:
    spans = [_node_span(node) for node in nodes]
    start = min(span[0] for span in spans)
    end = max(span[1] for span in spans)
    text = "".join(_slice(lines, s, e) for s, e in spans)
    return start, end, text


def _slice(lines: list[str], start: int, end: int) -> str:
    return "".join(lines[start - 1 : end])
