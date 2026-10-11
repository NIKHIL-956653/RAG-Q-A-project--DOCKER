"""Corpus discovery and loading, with an explicit report of everything that went wrong.

Fixes for the legacy loader (audit B2-B5):

* recursive discovery, so files in sub-folders are not skipped;
* sorted, platform-independent ordering, so chunk order is identical on every machine;
* empty or unreadable files and pages are recorded as ``IngestIssue`` instead of silently
  producing zero chunks;
* pypdf parser warnings, such as corrupt compressed streams, are captured per file.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .errors import IngestError
from .models import Document

TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

IssueKind = Literal[
    "empty_text", "extraction_error", "extraction_warnings", "unsupported_file", "decode_error"
]


@dataclass(frozen=True, slots=True)
class IngestIssue:
    source: str
    kind: IssueKind
    detail: str
    page: int | None = None


@dataclass(frozen=True, slots=True)
class Corpus:
    root: str
    files: tuple[str, ...]
    documents: tuple[Document, ...]
    issues: tuple[IngestIssue, ...]
    fingerprint: str  # sha256 over (doc_id, text) of every document, in order

    def document_map(self) -> dict[str, Document]:
        return {d.doc_id: d for d in self.documents}


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def discover_files(root: Path) -> tuple[list[Path], list[IngestIssue]]:
    """All supported files under ``root``, recursively, in sorted relative-path order.

    Hidden files and folders (names starting with ``.``) are ignored. Other unsupported
    files are reported as ``unsupported_file`` issues.
    """
    if not root.is_dir():
        raise IngestError(f"corpus directory not found: {root}")
    files: list[Path] = []
    issues: list[IngestIssue] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
        else:
            issues.append(
                IngestIssue(
                    rel.as_posix(), "unsupported_file", f"unsupported suffix {path.suffix!r}"
                )
            )
    return files, issues


def _load_text_file(
    path: Path, rel: str, min_chars: int
) -> tuple[list[Document], list[IngestIssue]]:
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return [], [IngestIssue(rel, "decode_error", f"not valid UTF-8: {exc}")]
    text = normalize_newlines(text)
    if len(text.strip()) < min_chars:
        return [], [IngestIssue(rel, "empty_text", "file contains no text")]
    return [Document(doc_id=rel, text=text, source=rel)], []


class _WarningCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _load_pdf(path: Path, rel: str, min_chars: int) -> tuple[list[Document], list[IngestIssue]]:
    from pypdf import PdfReader

    capture = _WarningCapture()
    pypdf_logger = logging.getLogger("pypdf")
    pypdf_logger.addHandler(capture)
    docs: list[Document] = []
    issues: list[IngestIssue] = []
    try:
        reader = PdfReader(str(path))
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = normalize_newlines(page.extract_text() or "")
            except Exception as exc:  # noqa: BLE001 - pypdf raises many types on corrupt pages
                issues.append(
                    IngestIssue(
                        rel, "extraction_error", f"{type(exc).__name__}: {exc}", page=number
                    )
                )
                continue
            if len(text.strip()) < min_chars:
                issues.append(
                    IngestIssue(
                        rel,
                        "empty_text",
                        "no extractable text (scanned image, blank or corrupt page)",
                        page=number,
                    )
                )
                continue
            docs.append(Document(doc_id=f"{rel}#page={number}", text=text, source=rel, page=number))
    except Exception as exc:  # noqa: BLE001 - unreadable file: report it, never crash the run
        return [], [IngestIssue(rel, "extraction_error", f"{type(exc).__name__}: {exc}")]
    finally:
        pypdf_logger.removeHandler(capture)
    if capture.messages:
        issues.append(
            IngestIssue(
                rel,
                "extraction_warnings",
                f"{len(capture.messages)} parser warning(s); first: {capture.messages[0][:200]}",
            )
        )
    return docs, issues


def corpus_fingerprint(documents: tuple[Document, ...]) -> str:
    digest = hashlib.sha256()
    for doc in documents:
        digest.update(doc.doc_id.encode("utf-8") + b"\x00")
        digest.update(hashlib.sha256(doc.text.encode("utf-8")).digest())
    return digest.hexdigest()


def load_corpus(root: Path, *, min_chars: int = 1) -> Corpus:
    """Load every supported file under ``root``. Never silently drops content."""
    files, issues = discover_files(root)
    documents: list[Document] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() in PDF_SUFFIXES:
            docs, file_issues = _load_pdf(path, rel, min_chars)
        else:
            docs, file_issues = _load_text_file(path, rel, min_chars)
        documents.extend(docs)
        issues.extend(file_issues)
    docs_tuple = tuple(documents)
    return Corpus(
        root=str(root),
        files=tuple(p.relative_to(root).as_posix() for p in files),
        documents=docs_tuple,
        issues=tuple(issues),
        fingerprint=corpus_fingerprint(docs_tuple),
    )
