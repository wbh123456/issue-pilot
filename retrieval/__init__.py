"""Host-side code retrieval (AST chunks, BM25, dense, RRF). Not a V0/V1 tool."""

from retrieval.chunker import Chunk, chunk_file, chunk_repo
from retrieval.dense import FastEmbedEmbedder, HashingEmbedder
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.indexer import CodeIndex, build_index

__all__ = [
    "Chunk",
    "CodeIndex",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "build_index",
    "chunk_file",
    "chunk_repo",
    "reciprocal_rank_fusion",
]
