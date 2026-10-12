"""Okapi BM25, implemented in pure Python so it is deterministic and dependency-free.

score(q, d) = sum over unique query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))      (Lucene variant, always > 0)

Tokens come from ``text.retrieval_tokens``: lowercased, stopwords removed, no stemming.
Only chunks with a positive score are returned, so a query with no term overlap gets
no passages.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ..errors import ConfigError
from ..models import Chunk, ScoredChunk
from ..text import retrieval_tokens
from . import rank


class BM25Retriever:
    name = "bm25"
    score_kind = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        if k1 < 0:
            raise ConfigError(f"bm25.k1 must be >= 0, got {k1}")
        if not 0.0 <= b <= 1.0:
            raise ConfigError(f"bm25.b must be within [0, 1], got {b}")
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 0.0
        self._postings: dict[str, list[int]] = {}

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tfs = [Counter(retrieval_tokens(c.text)) for c in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._tfs]
        n = len(self._chunks)
        self._avgdl = (sum(self._lengths) / n) if n else 0.0
        df: Counter[str] = Counter()
        postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self._tfs):
            for term in tf:
                df[term] += 1
                postings.setdefault(term, []).append(i)
        self._postings = postings
        self._idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score_all(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = {}
        if not self._chunks or self._avgdl == 0.0:
            return scores
        for term in dict.fromkeys(retrieval_tokens(query)):  # unique, order-preserving
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in self._postings[term]:
                tf = self._tfs[i][term]
                norm = self.k1 * (1.0 - self.b + self.b * self._lengths[i] / self._avgdl)
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1.0) / (tf + norm)
        return scores

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        if k < 1:
            raise ConfigError(f"k must be >= 1, got {k}")
        scores = self.score_all(query)
        return rank([(s, self._chunks[i]) for i, s in scores.items() if s > 0.0], k)

    def describe(self) -> dict[str, object]:
        return {"method": "bm25", "k1": self.k1, "b": self.b, "tokenizer": "retrieval_tokens/v1"}
