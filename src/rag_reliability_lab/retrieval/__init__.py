"""Retrievers rank chunks for a query. All implementations break ties deterministically."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..models import Chunk, ScoredChunk


class Retriever(Protocol):
    """Index a fixed set of chunks once, then answer ranked top-k queries."""

    name: str
    score_kind: str  # how to read ScoredChunk.score, e.g. "bm25" or "neg_squared_l2"

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...

    def describe(self) -> dict[str, object]: ...


def tie_break_key(chunk: Chunk) -> tuple[str, int]:
    """Equal scores are ordered by corpus position (document id, then chunk index)."""
    return (chunk.doc_id, chunk.index)


def rank(scored: list[tuple[float, Chunk]], k: int) -> list[ScoredChunk]:
    ordered = sorted(scored, key=lambda item: (-item[0], tie_break_key(item[1])))
    return [ScoredChunk(chunk=c, score=s, rank=i) for i, (s, c) in enumerate(ordered[:k], start=1)]
