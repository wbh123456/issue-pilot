"""In-memory hybrid index over ``app/**/*.py`` of one benchmark worktree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.context import RETRIEVE_K
from retrieval.chunker import Chunk, chunk_repo
from retrieval.dense import DenseIndex, Embedder, HashingEmbedder
from retrieval.hybrid import hybrid_search
from retrieval.lexical import BM25Index


@dataclass
class CodeIndex:
    repo_root: Path
    chunks: list[Chunk]
    embedder: Embedder
    bm25: BM25Index
    dense: DenseIndex

    def search_bm25(self, query: str, k: int = RETRIEVE_K) -> list[Chunk]:
        return self.bm25.search(query, k=k)

    def search_dense(self, query: str, k: int = RETRIEVE_K) -> list[Chunk]:
        return self.dense.search(query, k=k)

    def search_hybrid(self, query: str, k: int = RETRIEVE_K) -> list[Chunk]:
        return hybrid_search(query, self.bm25, self.dense, k=k)


def build_index(
    repo_path: str | Path,
    *,
    embedder: Embedder | None = None,
) -> CodeIndex:
    """Build BM25 + dense indexes. Default embedder is hashing (no download)."""
    chosen = embedder if embedder is not None else HashingEmbedder()
    chunks = chunk_repo(repo_path)
    return CodeIndex(
        repo_root=Path(repo_path).resolve(),
        chunks=chunks,
        embedder=chosen,
        bm25=BM25Index(chunks),
        dense=DenseIndex(chunks, chosen),
    )
