"""Retrieve node: hybrid code search. Deterministic, 0 LLM calls."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from harness.context import MAX_PLANNER_CONTEXT_CHARS, RETRIEVE_K, truncate_chars
from retrieval.chunker import Chunk
from retrieval.dense import Embedder, FastEmbedEmbedder, HashingEmbedder
from retrieval.indexer import build_index

from ._runtime import merge_telemetry, require_config


def _embedder_from_config(cfg: dict) -> Embedder:
    if cfg.get("embedder") is not None:
        return cfg["embedder"]
    name = str(cfg.get("embedder_name") or "hashing").strip().lower()
    if name in {"fastembed", "bge"}:
        return FastEmbedEmbedder()
    return HashingEmbedder()


def _query(state: AgentState) -> str:
    issue = (state.get("issue") or "").strip()
    analysis = (state.get("analysis") or "").strip()
    if analysis:
        return f"{issue}\n{analysis}"
    return issue


def _unique_paths(chunks: Sequence[Chunk]) -> list[str]:
    seen: list[str] = []
    known: set[str] = set()
    for chunk in chunks:
        path = chunk.path.replace("\\", "/")
        if not path or path in known:
            continue
        known.add(path)
        seen.append(path)
    return seen


def format_retrieved_context(chunks: Sequence[Chunk]) -> str:
    """Bounded snippet blob for planner / executor (mechanical truncation)."""
    parts: list[str] = []
    for chunk in chunks:
        header = (
            f"### {chunk.path}  {chunk.symbol} ({chunk.type})  "
            f"L{chunk.start_line}-{chunk.end_line}"
        )
        parts.append(f"{header}\n{chunk.text}")
    blob = "\n\n".join(parts) if parts else ""
    return truncate_chars(
        blob, MAX_PLANNER_CONTEXT_CHARS, label="MAX_PLANNER_CONTEXT_CHARS"
    )


def retrieve_context(state: AgentState, config: RunnableConfig) -> dict:
    """Hybrid search on issue + analysis. No LLM."""
    cfg = require_config(config, "repo_path")
    repo_path = str(cfg["repo_path"])
    k = int(cfg.get("retrieve_k") or RETRIEVE_K)
    index = build_index(repo_path, embedder=_embedder_from_config(cfg))
    chunks = index.search_hybrid(_query(state), k=k)
    files = _unique_paths(chunks)
    blob = format_retrieved_context(chunks)
    telemetry = merge_telemetry(
        state,
        retrieval_calls=1,
        stage_tokens={
            "retrieve": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "llm_calls": 0,
            }
        },
    )
    return {
        "relevant_files": files,
        "retrieved_context": blob,
        "telemetry": telemetry,
        "status": "retrieved",
    }
