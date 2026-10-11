"""Recursive character chunking with exact character offsets.

``RecursiveCharacterSplitter`` re-implements LangChain's ``RecursiveCharacterTextSplitter``
with its default settings: separators ``["\\n\\n", "\\n", " ", ""]``, ``keep_separator=True``
(separator kept at the start of the next piece), ``strip_whitespace=True`` and
``length_function=len``. This is the chunker the legacy app used (1000/200). A parity test
checks that both produce identical chunk texts, so the baseline can be reproduced without
depending on LangChain.

Offsets are tracked through the algorithm rather than recovered afterwards. LangChain's
``add_start_index`` searches for each chunk's text from an estimated position. When a
chunk is shorter than the overlap that search can land on an earlier duplicate, giving
offsets that run backwards. Evidence matching depends on offsets, so here they are exact by
construction and verified: ``document.text[start:end] == chunk.text``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from .errors import ConfigError
from .models import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ", "")

Piece = tuple[str, int]  # (text, absolute start offset)


class RecursiveCharacterSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ConfigError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ConfigError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ConfigError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            )
        if not separators:
            raise ConfigError("separators must not be empty")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators: tuple[str, ...] = tuple(separators)

    # --- public API -------------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        return [piece for piece, _ in self.split_with_offsets(text)]

    def split_with_offsets(self, text: str) -> list[Piece]:
        return self._split(text, 0, list(self.separators))

    # --- LangChain-equivalent internals, carrying offsets ------------------------------------
    @staticmethod
    def _split_keep_start(text: str, separator: str) -> list[Piece]:
        """Split keeping each separator at the start of the following piece. The pieces
        concatenate back to ``text``, so cumulative lengths give exact offsets."""
        if separator:
            parts = re.split(f"({re.escape(separator)})", text)
            pieces = [parts[i] + parts[i + 1] for i in range(1, len(parts), 2)]
            pieces = [parts[0], *pieces]
        else:
            pieces = list(text)
        out: list[Piece] = []
        position = 0
        for piece in pieces:
            if piece:
                out.append((piece, position))
            position += len(piece)
        return out

    def _split(self, text: str, base: int, separators: list[str]) -> list[Piece]:
        final_chunks: list[Piece] = []
        separator = separators[-1]
        new_separators: list[str] = []
        for i, candidate in enumerate(separators):
            if not candidate:
                separator = candidate
                break
            if candidate in text:
                separator = candidate
                new_separators = separators[i + 1 :]
                break

        good_splits: list[Piece] = []
        for piece, local in self._split_keep_start(text, separator):
            start = base + local
            if len(piece) < self.chunk_size:
                good_splits.append((piece, start))
                continue
            if good_splits:
                final_chunks.extend(self._merge(good_splits))
                good_splits = []
            if not new_separators:
                final_chunks.append((piece, start))  # LangChain keeps this piece unstripped
            else:
                final_chunks.extend(self._split(piece, start, new_separators))
        if good_splits:
            final_chunks.extend(self._merge(good_splits))
        return final_chunks

    @staticmethod
    def _join(current: list[Piece]) -> Piece | None:
        # Pieces in a merge window are contiguous, so the joined text starts at the first one.
        text = "".join(piece for piece, _ in current)
        stripped = text.strip()
        if not stripped:
            return None
        return stripped, current[0][1] + (len(text) - len(text.lstrip()))

    def _merge(self, splits: Iterable[Piece]) -> list[Piece]:
        # LangChain's _merge_splits with separator "" (keep_separator=True always merges
        # with an empty separator, so every separator-length term is zero).
        docs: list[Piece] = []
        current: list[Piece] = []
        total = 0
        for piece, start in splits:
            piece_len = len(piece)
            if current and total + piece_len > self.chunk_size:
                joined = self._join(current)
                if joined is not None:
                    docs.append(joined)
                while total > self.chunk_overlap or (
                    total + piece_len > self.chunk_size and total > 0
                ):
                    total -= len(current[0][0])
                    current = current[1:]
            current.append((piece, start))
            total += piece_len
        joined = self._join(current)
        if joined is not None:
            docs.append(joined)
        return docs


def chunk_documents(
    documents: Iterable[Document], splitter: RecursiveCharacterSplitter
) -> list[Chunk]:
    """Split documents into chunks with stable ids (``<doc_id>::c0000``) and exact offsets."""
    chunks: list[Chunk] = []
    for doc in documents:
        for i, (piece, start) in enumerate(splitter.split_with_offsets(doc.text)):
            end = start + len(piece)
            if doc.text[start:end] != piece:  # invariant; never expected to fail
                raise AssertionError(f"chunk {i} of {doc.doc_id} does not match its offsets")
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{i:04d}",
                    doc_id=doc.doc_id,
                    index=i,
                    start=start,
                    end=end,
                    text=piece,
                )
            )
    return chunks
