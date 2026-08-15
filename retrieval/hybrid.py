"""Reciprocal Rank Fusion over lexical + dense rankings."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from harness.context import RETRIEVE_K, RRF_K
from retrieval.chunker import Chunk
from retrieval.dense import DenseIndex
from retrieval.lexical import BM25Index


def reciprocal_rank_fusion(
    ranked_ids: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """RRF: ``score(d) = sum 1 / (k + rank_i(d))``. Ties break on id."""
    if k < 1:
        raise ValueError("k must be >= 1")
    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_ids:
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def fuse_chunk_lists(
    rankings: Sequence[Sequence[Chunk]],
    *,
    k: int = RETRIEVE_K,
    rrf_k: int = RRF_K,
) -> list[Chunk]:
    ranked_ids = [[chunk.id for chunk in ranking] for ranking in rankings]
    fused = reciprocal_rank_fusion(ranked_ids, k=rrf_k)
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for chunk in ranking:
            by_id.setdefault(chunk.id, chunk)
    out: list[Chunk] = []
    for doc_id, _score in fused:
        chunk = by_id.get(doc_id)
        if chunk is None:
            continue
        out.append(chunk)
        if len(out) >= k:
            break
    return out


def hybrid_search(
    query: str,
    bm25: BM25Index,
    dense: DenseIndex,
    k: int = RETRIEVE_K,
    rrf_k: int = RRF_K,
) -> list[Chunk]:
    pool = max(len(bm25.chunks), len(dense.chunks), k)
    return fuse_chunk_lists(
        [
            bm25.search(query, k=pool),
            dense.search(query, k=pool),
        ],
        k=k,
        rrf_k=rrf_k,
    )
