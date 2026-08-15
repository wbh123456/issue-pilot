"""Dense retrieval: pluggable embedder + numpy cosine. No FAISS/Chroma."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from harness.context import RETRIEVE_K
from retrieval.chunker import Chunk, index_text
from retrieval.tokenize import tokenize


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float matrix."""


class HashingEmbedder:
    """Deterministic signed hashing vectorizer. No network, no model download."""

    def __init__(self, dim: int = 256):
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        rows = [self._embed_one(text) for text in texts]
        return np.stack(rows, axis=0)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            n = int.from_bytes(digest, "little")
            idx = n % self.dim
            sign = 1.0 if ((n // self.dim) % 2 == 0) else -1.0
            vec[idx] += sign
        return vec


class FastEmbedEmbedder:
    """ONNX embedder for live retrieval eval. Do not construct in default pytest."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        model = self._load()
        vectors = list(model.embed(list(texts)))
        return np.asarray(vectors, dtype=np.float64)

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "fastembed is required for FastEmbedEmbedder; "
                    "use HashingEmbedder in unit tests"
                ) from exc
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model


def cosine_similarity(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q = np.asarray(queries, dtype=np.float64)
    c = np.asarray(corpus, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    cn = np.linalg.norm(c, axis=1, keepdims=True)
    q = q / np.clip(qn, 1e-12, None)
    c = c / np.clip(cn, 1e-12, None)
    return q @ c.T


class DenseIndex:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder):
        self.chunks = list(chunks)
        self.embedder = embedder
        if self.chunks:
            texts = [index_text(chunk) for chunk in self.chunks]
            self._matrix = np.asarray(embedder.embed(texts), dtype=np.float64)
        else:
            self._matrix = np.zeros((0, 1), dtype=np.float64)

    def search(self, query: str, k: int = RETRIEVE_K) -> list[Chunk]:
        if not self.chunks or not query.strip() or k <= 0:
            return []
        q = np.asarray(self.embedder.embed([query]), dtype=np.float64)
        sims = cosine_similarity(q, self._matrix)[0]
        order = np.argsort(-sims, kind="stable")
        return [self.chunks[int(i)] for i in order[:k]]
