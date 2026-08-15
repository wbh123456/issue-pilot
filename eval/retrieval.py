"""Retrieval eval: grep vs BM25 vs dense vs hybrid. No LLM, no V2 graph.

Query is raw ``dataset.issue`` text. File-level Recall@K is the metric.
Grep uses the existing ``grep_code`` tool (substring + hit counts). It must
not call BM25, or the ablation is meaningless.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from agent.tools.search import grep_code
from eval.metrics import recall_at_k, unique_paths
from eval.repository import reset_repo
from harness.context import RETRIEVE_K
from retrieval.dense import Embedder, FastEmbedEmbedder, HashingEmbedder
from retrieval.indexer import CodeIndex, build_index

RetrievalMode = Literal["grep", "bm25", "dense", "hybrid"]
INDEX_MODES: tuple[RetrievalMode, ...] = ("bm25", "dense", "hybrid")
ALL_MODES: tuple[RetrievalMode, ...] = ("grep", *INDEX_MODES)

HARNESS_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = HARNESS_ROOT / "runs"
DATASET_PATH = HARNESS_ROOT / "eval" / "dataset.json"

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_MIN_GREP_TOKEN_LEN = 3
# Function words dropped so grep is not a bag of "the"/"with" hits.
_GREP_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "get",
        "had",
        "has",
        "have",
        "i",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "out",
        "post",
        "she",
        "so",
        "than",
        "that",
        "the",
        "then",
        "this",
        "to",
        "until",
        "was",
        "were",
        "when",
        "with",
    }
)


def make_embedder(name: str) -> Embedder:
    key = (name or "hashing").strip().lower()
    if key in {"hashing", "hash"}:
        return HashingEmbedder()
    if key in {"fastembed", "bge"}:
        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder {name!r}; use 'hashing' or 'fastembed'")


def grep_query_tokens(issue: str) -> list[str]:
    """Issue tokens for the grep baseline. Original case; no camel/snake split."""
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in _IDENT.findall(issue or ""):
        key = raw.lower()
        if len(raw) < _MIN_GREP_TOKEN_LEN or key in _GREP_STOPWORDS or key in seen:
            continue
        seen.add(key)
        tokens.append(raw)
    return tokens


def rank_files_by_grep(repo_path: str | Path, issue: str) -> list[str]:
    """Rank files by ``grep_code`` hit count. Not BM25."""
    counts: dict[str, int] = defaultdict(int)
    for token in grep_query_tokens(issue):
        raw = grep_code(repo_path, token)
        if not raw or raw.startswith("Error:") or raw.strip() == "(no matches)":
            continue
        for line in raw.splitlines():
            path = _grep_hit_path(line)
            if path:
                counts[path] += 1
    return sorted(counts, key=lambda path: (-counts[path], path))


def _grep_hit_path(line: str) -> str | None:
    # grep_code format: "{rel}:{lineno}:{text}" with posix rel paths.
    parts = line.split(":", 2)
    if len(parts) < 2:
        return None
    path = parts[0].replace("\\", "/").strip()
    return path or None


def rank_files_by_index(
    index: CodeIndex,
    mode: RetrievalMode,
    query: str,
    k: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Collapse chunk hits to unique paths, preserving rank."""
    pool = max(len(index.chunks), k)
    if mode == "bm25":
        chunks = index.search_bm25(query, k=pool)
    elif mode == "dense":
        chunks = index.search_dense(query, k=pool)
    elif mode == "hybrid":
        chunks = index.search_hybrid(query, k=pool)
    else:
        raise ValueError(f"not an index mode: {mode}")

    files: list[str] = []
    details: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.path in seen:
            continue
        seen.add(chunk.path)
        files.append(chunk.path)
        details.append(
            {
                "path": chunk.path,
                "symbol": chunk.symbol,
                "type": chunk.type,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
        )
        if len(files) >= k:
            break
    return files, details


def evaluate_repo(
    task: dict[str, Any],
    repo_path: str | Path,
    *,
    index: CodeIndex | None = None,
    embedder: Embedder | None = None,
    k: int = RETRIEVE_K,
) -> dict[str, Any]:
    """Run four retrieval modes on an already-reset worktree. No LLM."""
    if k < 1:
        raise ValueError("k must be >= 1")
    repo = Path(repo_path)
    built = index or build_index(repo, embedder=embedder or HashingEmbedder())
    query = str(task.get("issue") or "")
    expected = list(task.get("expected_files") or [])

    grep_ranked = rank_files_by_grep(repo, query)
    grep_files = unique_paths(grep_ranked, k=k)
    modes: dict[str, Any] = {
        "grep": {
            "retrieved_files": grep_files,
            "recall_at_k": recall_at_k(grep_files, expected, k=k),
            "tokens": grep_query_tokens(query),
        }
    }
    for mode in INDEX_MODES:
        files, chunks = rank_files_by_index(built, mode, query, k)
        modes[mode] = {
            "retrieved_files": files,
            "recall_at_k": recall_at_k(files, expected, k=k),
            "chunks": chunks,
        }

    return {
        "task_id": task["id"],
        "split": task.get("split"),
        "issue": query,
        "expected_files": expected,
        "k": k,
        "embedder": type(built.embedder).__name__,
        "chunk_count": len(built.chunks),
        "modes": modes,
    }


def save_retrieve_run(record: dict[str, Any]) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task_id = record.get("task_id") or "split"
    out = RUNS_DIR / f"{task_id}-retrieve-{stamp}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def run_retrieval_eval(
    *,
    task_id: str | None = None,
    split: str | None = None,
    k: int = RETRIEVE_K,
    embedder_name: str = "hashing",
    reset: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Reset (optional), index once per commit, score selected tasks."""
    tasks = _select_tasks(task_id=task_id, split=split)
    embedder = make_embedder(embedder_name)

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str]] = []
    for task in tasks:
        repo = _resolve_repo_path(task)
        key = (str(repo), str(task["base_commit"]))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(task)

    records: list[dict[str, Any]] = []
    for key in order:
        repo = Path(key[0])
        commit = key[1]
        if reset:
            reset_repo(repo, commit)
        index = build_index(repo, embedder=embedder)
        for task in groups[key]:
            record = evaluate_repo(task, repo, index=index, k=k)
            if save:
                record["run_path"] = str(save_retrieve_run(record))
            records.append(record)

    means = {
        mode: (
            sum(float(row["modes"][mode]["recall_at_k"]) for row in records)
            / len(records)
            if records
            else 0.0
        )
        for mode in ALL_MODES
    }
    return {
        "split": split,
        "k": k,
        "embedder": type(embedder).__name__,
        "tasks": records,
        "mean_recall_at_k": means,
    }


def _load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset.json must be a JSON array of tasks")
    return data


def _get_task(task_id: str, path: Path = DATASET_PATH) -> dict[str, Any]:
    for task in _load_dataset(path):
        if task.get("id") == task_id:
            return task
    known = ", ".join(t["id"] for t in _load_dataset(path))
    raise KeyError(f"unknown task_id={task_id!r}; known: {known}")


def _resolve_repo_path(task: dict[str, Any]) -> Path:
    raw = Path(task["repo_path"])
    repo = raw if raw.is_absolute() else (HARNESS_ROOT / raw).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"benchmark repo not found: {repo}")
    return repo


def _select_tasks(
    *,
    task_id: str | None,
    split: str | None,
) -> list[dict[str, Any]]:
    if task_id and split:
        task = _get_task(task_id)
        if task.get("split") != split:
            raise ValueError(
                f"{task_id} is split {task.get('split')!r}, not {split!r}"
            )
        return [task]
    if task_id:
        return [_get_task(task_id)]
    if split:
        tasks = [t for t in _load_dataset() if t.get("split") == split]
        if not tasks:
            raise ValueError(f"no tasks in split {split!r}")
        return tasks
    raise ValueError("pass a task_id or --split")
