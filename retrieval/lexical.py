"""BM25 over AST chunks (not whole files)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from harness.context import RETRIEVE_K
from retrieval.chunker import Chunk, index_text
from retrieval.tokenize import tokenize


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        tokenized = [tokenize(index_text(chunk)) or ["_empty"] for chunk in self.chunks]
        self._bm25 = BM25Okapi(tokenized) if self.chunks else None

    def search(self, query: str, k: int = RETRIEVE_K) -> list[Chunk]:
        if self._bm25 is None or not query.strip() or k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        return [self.chunks[int(i)] for i in order[:k]]
