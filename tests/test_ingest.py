from __future__ import annotations

from pathlib import Path

import pytest

from rag_reliability_lab.errors import IngestError
from rag_reliability_lab.ingest import discover_files, load_corpus

from .conftest import make_pdf


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestError):
        load_corpus(tmp_path / "nope")


def test_discovery_is_recursive_sorted_and_skips_hidden(tmp_path: Path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / "b" / "z.md").write_text("z")
    (tmp_path / "a" / "y.txt").write_text("y")
    (tmp_path / "top.MD").write_text("upper-case suffix")
    (tmp_path / ".git" / "config.md").write_text("hidden")
    (tmp_path / ".secret.md").write_text("hidden")
    (tmp_path / "data.csv").write_text("x,y")
    files, issues = discover_files(tmp_path)
    assert [f.relative_to(tmp_path).as_posix() for f in files] == ["a/y.txt", "b/z.md", "top.MD"]
    assert [(i.source, i.kind) for i in issues] == [("data.csv", "unsupported_file")]


def test_text_is_newline_normalised_and_bom_stripped(tmp_path: Path) -> None:
    (tmp_path / "w.md").write_bytes("﻿line one\r\nline two\rline three".encode())
    corpus = load_corpus(tmp_path)
    assert corpus.documents[0].text == "line one\nline two\nline three"


def test_empty_and_undecodable_files_are_reported_not_dropped(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("   \n")
    (tmp_path / "latin1.txt").write_bytes("caf\xe9".encode("latin-1"))
    corpus = load_corpus(tmp_path)
    assert corpus.documents == ()
    assert sorted((i.source, i.kind) for i in corpus.issues) == [
        ("empty.md", "empty_text"),
        ("latin1.txt", "decode_error"),
    ]


def test_pdf_pages_become_documents_and_blank_pages_are_reported(tmp_path: Path) -> None:
    make_pdf(tmp_path / "doc.pdf", ["Hello from page one", "", "Page three text"])
    corpus = load_corpus(tmp_path)
    assert [d.doc_id for d in corpus.documents] == ["doc.pdf#page=1", "doc.pdf#page=3"]
    assert "Hello from page one" in corpus.documents[0].text
    assert [(i.kind, i.page) for i in corpus.issues] == [("empty_text", 2)]


def test_corrupt_pdf_is_reported_without_crashing(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4\nthis is not a pdf at all")
    (tmp_path / "ok.md").write_text("still loaded")
    corpus = load_corpus(tmp_path)
    assert [d.doc_id for d in corpus.documents] == ["ok.md"]
    assert any(i.source == "broken.pdf" and i.kind == "extraction_error" for i in corpus.issues)


def test_fingerprint_changes_with_content(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("one")
    first = load_corpus(tmp_path).fingerprint
    assert load_corpus(tmp_path).fingerprint == first
    (tmp_path / "a.md").write_text("two")
    assert load_corpus(tmp_path).fingerprint != first
