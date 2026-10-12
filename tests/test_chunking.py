from __future__ import annotations

import random

import pytest

from rag_reliability_lab.chunking import RecursiveCharacterSplitter, chunk_documents
from rag_reliability_lab.errors import ConfigError
from rag_reliability_lab.ingest import load_corpus
from rag_reliability_lab.models import Document

from .conftest import DATASET_DIR

CONFIGS = [(1000, 200), (500, 100), (300, 50), (200, 0), (64, 16), (10, 3)]


def _random_texts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = ["a", "word", "longerword", " ", " ", "\n", "\n\n", ".", "x" * 30, "\t", "  "]
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 600))) for _ in range(n)]


def _sample_texts() -> list[str]:
    corpus = load_corpus(DATASET_DIR / "corpus")
    return [d.text for d in corpus.documents] + _random_texts(120, seed=7)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_parameters_rejected(size: int, overlap: int) -> None:
    with pytest.raises(ConfigError):
        RecursiveCharacterSplitter(size, overlap)


@pytest.mark.parametrize(("size", "overlap"), CONFIGS)
def test_chunks_are_exact_substrings_with_monotonic_offsets(size: int, overlap: int) -> None:
    splitter = RecursiveCharacterSplitter(size, overlap)
    docs = [Document(doc_id=f"d{i}", text=t, source="x") for i, t in enumerate(_sample_texts())]
    chunks = chunk_documents(docs, splitter)
    by_doc: dict[str, list[int]] = {}
    texts = {d.doc_id: d.text for d in docs}
    for c in chunks:
        assert texts[c.doc_id][c.start : c.end] == c.text
        assert c.chunk_id == f"{c.doc_id}::c{c.index:04d}"
        by_doc.setdefault(c.doc_id, []).append(c.start)
    for starts in by_doc.values():
        assert starts == sorted(starts)


def test_chunking_is_deterministic() -> None:
    docs = load_corpus(DATASET_DIR / "corpus").documents
    a = chunk_documents(docs, RecursiveCharacterSplitter(300, 50))
    b = chunk_documents(docs, RecursiveCharacterSplitter(300, 50))
    assert a == b


def test_empty_and_whitespace_documents_yield_no_chunks() -> None:
    splitter = RecursiveCharacterSplitter(100, 10)
    assert splitter.split_text("") == []
    assert splitter.split_text("   \n\n  ") == []


@pytest.mark.parity
@pytest.mark.parametrize(("size", "overlap"), CONFIGS)
def test_parity_with_langchain(size: int, overlap: int) -> None:
    """The legacy baseline used LangChain's splitter. Chunk texts must be identical."""
    splitters = pytest.importorskip("langchain_text_splitters")
    reference = splitters.RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    ours = RecursiveCharacterSplitter(size, overlap)
    for text in _sample_texts():
        assert ours.split_text(text) == reference.split_text(text)


@pytest.mark.parity
@pytest.mark.parametrize(("size", "overlap"), [(1000, 200), (500, 100), (300, 50)])
def test_offsets_agree_with_langchain_on_real_text(size: int, overlap: int) -> None:
    """On real documents LangChain's search-based start_index is right, and our exact offsets
    agree with it. (On degenerate inputs its search can run backwards; ours cannot.)"""
    splitters = pytest.importorskip("langchain_text_splitters")
    reference = splitters.RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True
    )
    docs = load_corpus(DATASET_DIR / "corpus").documents
    ref = reference.create_documents([d.text for d in docs])
    ours = chunk_documents(docs, RecursiveCharacterSplitter(size, overlap))
    assert [d.metadata["start_index"] for d in ref] == [c.start for c in ours]
