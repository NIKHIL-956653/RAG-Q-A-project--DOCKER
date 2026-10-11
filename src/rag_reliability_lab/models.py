"""Core data types shared across ingestion, retrieval, generation and scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    """A unit of source text.

    Text files yield one document. PDFs yield one document per page, which mirrors
    the legacy ``PyPDFLoader`` behaviour, so chunks never span pages.
    """

    doc_id: str  # e.g. "handbook/policy.md" or "units/unit-03.pdf#page=4"
    text: str  # newline-normalised (LF only); evidence offsets refer to this text
    source: str  # file path relative to the corpus root, POSIX separators
    page: int | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage. ``text == document.text[start:end]`` always holds."""

    chunk_id: str
    doc_id: str
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk returned by a retriever. ``rank`` is 1-based; a higher ``score`` is better."""

    chunk: Chunk
    score: float
    rank: int
