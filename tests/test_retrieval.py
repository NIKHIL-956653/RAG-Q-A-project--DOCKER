from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from rag_reliability_lab.errors import ConfigError
from rag_reliability_lab.models import Chunk
from rag_reliability_lab.retrieval.bm25 import BM25Retriever


def _chunk(doc: str, index: int, text: str) -> Chunk:
    return Chunk(f"{doc}::c{index:04d}", doc, index, 0, len(text), text)


CHUNKS = [
    _chunk("a", 0, "red apple pie"),
    _chunk("b", 0, "green apple"),
    _chunk("c", 0, "banana bread and banana cake"),
]


def test_bm25_matches_hand_computed_scores() -> None:
    r = BM25Retriever(k1=1.2, b=0.75)
    r.index(CHUNKS)
    # lengths after stopword removal: 3, 2, 4 tokens ("and" is a stopword); avgdl = 3
    n, df_apple = 3, 2
    idf = math.log(1 + (n - df_apple + 0.5) / (df_apple + 0.5))

    def term(tf: int, dl: int) -> float:
        return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / 3))

    scores = r.score_all("apple")
    assert scores[0] == pytest.approx(term(1, 3))
    assert scores[1] == pytest.approx(term(1, 2))
    ranked = r.retrieve("apple", 3)
    assert [s.chunk.doc_id for s in ranked] == ["b", "a"]  # shorter doc wins; c has no match
    assert [s.rank for s in ranked] == [1, 2]


def test_bm25_returns_nothing_without_term_overlap() -> None:
    r = BM25Retriever()
    r.index(CHUNKS)
    assert r.retrieve("zebra", 3) == []
    assert r.retrieve("the of and", 3) == []  # only stopwords


def test_bm25_ties_break_by_corpus_position() -> None:
    chunks = [
        _chunk("z", 0, "widget text"),
        _chunk("a", 1, "widget text"),
        _chunk("a", 0, "widget text"),
    ]
    r = BM25Retriever()
    r.index(chunks)
    assert [s.chunk.chunk_id for s in r.retrieve("widget", 3)] == [
        "a::c0000",
        "a::c0001",
        "z::c0000",
    ]


def test_bm25_validates_parameters() -> None:
    with pytest.raises(ConfigError):
        BM25Retriever(b=1.5)
    r = BM25Retriever()
    r.index(CHUNKS)
    with pytest.raises(ConfigError):
        r.retrieve("apple", 0)


def test_bm25_empty_index() -> None:
    r = BM25Retriever()
    r.index([])
    assert r.retrieve("apple", 3) == []


class FakeEmbedder:
    """Deterministic 3-dimensional embeddings keyed by vocabulary."""

    name = "fake"
    VOCAB = ("apple", "banana", "cherry")

    def embed(self, texts: Sequence[str]) -> Any:
        np = pytest.importorskip("numpy")
        return np.array([[t.count(w) for w in self.VOCAB] for t in texts], dtype=np.float32)


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
def test_dense_retriever_ranks_and_breaks_ties(metric: str) -> None:
    pytest.importorskip("numpy")
    from rag_reliability_lab.retrieval.dense import DenseRetriever

    chunks = [
        _chunk("x", 0, "banana banana"),
        _chunk("y", 0, "apple"),
        _chunk("a", 0, "apple"),  # identical to y: tie broken by doc id
        _chunk("z", 0, "cherry"),
    ]
    r = DenseRetriever(FakeEmbedder(), metric=metric)  # type: ignore[arg-type]
    r.index(chunks)
    top = r.retrieve("apple", 2)
    assert [s.chunk.chunk_id for s in top] == ["a::c0000", "y::c0000"]
    assert top[0].score >= top[1].score
    assert r.describe() == {"method": "dense", "model": "fake", "metric": metric}


def test_dense_retriever_rejects_unknown_metric() -> None:
    from rag_reliability_lab.retrieval.dense import DenseRetriever

    with pytest.raises(ConfigError):
        DenseRetriever(FakeEmbedder(), metric="manhattan")  # type: ignore[arg-type]
