"""Identifier-aware tokenizer for BM25 and the hashing embedder."""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text: str) -> list[str]:
    """Lowercased identifiers, plus snake_case / CamelCase pieces."""
    tokens: list[str] = []
    for raw in _IDENT.findall(text):
        lower = raw.lower()
        tokens.append(lower)
        if "_" in raw:
            tokens.extend(part.lower() for part in raw.split("_") if part)
            continue
        pieces = _CAMEL.findall(raw)
        if len(pieces) > 1:
            tokens.extend(part.lower() for part in pieces)
    return tokens
