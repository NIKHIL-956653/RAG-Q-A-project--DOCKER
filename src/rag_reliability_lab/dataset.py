"""Evaluation dataset: schema, loading and validation.

A dataset directory contains:

    dataset.toml       name, version, description, corpus dir, questions file
    corpus/            the documents (.md, .txt or .pdf)
    questions.jsonl    one question per line (schema below)

Evidence is annotated at **span level** (an exact quote inside a document), not per chunk.
That keeps the labels independent of the chunking configuration, so the same dataset can
score any chunk size.

Question schema (unknown keys are rejected):

    id               str, unique, [a-z0-9_-]
    question         str
    category         one of CATEGORIES
    answerable       bool
    evidence         [{"doc": <doc_id>, "quote": <exact text>}], every item required
    reference_answer str (answerable only)
    required_facts   [{"id": str, "any_of": [str, ...], "derived": bool?}] (answerable only)
    stale_facts      [{"id": str, "any_of": [str, ...]}] outdated values that must not be
                     given as the answer
    absent_terms     [str], phrases that must NOT occur anywhere in the corpus (proves an
                     unanswerable question really is unanswerable)
    notes            str
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DatasetError
from .ingest import Corpus, load_corpus, normalize_newlines
from .text import contains_phrase

CATEGORIES: tuple[str, ...] = (
    "factual",
    "multi_passage",
    "unanswerable",
    "distractor",
    "conflicting",
    "poorly_phrased",
    "attribution",
)
_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_QUESTION_KEYS = frozenset(
    {
        "id",
        "question",
        "category",
        "answerable",
        "evidence",
        "reference_answer",
        "required_facts",
        "stale_facts",
        "absent_terms",
        "notes",
    }
)
_MIN_QUOTE_CHARS = 15


@dataclass(frozen=True, slots=True)
class Fact:
    fact_id: str
    any_of: tuple[str, ...]
    derived: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    doc_id: str
    quote: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Question:
    qid: str
    question: str
    category: str
    answerable: bool
    evidence: tuple[EvidenceSpan, ...]
    reference_answer: str
    required_facts: tuple[Fact, ...]
    stale_facts: tuple[Fact, ...]
    absent_terms: tuple[str, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class Dataset:
    name: str
    version: str
    description: str
    root: str
    corpus: Corpus
    questions: tuple[Question, ...]
    questions_sha256: str


def _parse_facts(
    raw: Any, where: str, problems: list[str], *, allow_derived: bool
) -> tuple[Fact, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        problems.append(f"{where}: must be a list")
        return ()
    facts: list[Fact] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        loc = f"{where}[{i}]"
        allowed = {"id", "any_of", "derived"} if allow_derived else {"id", "any_of"}
        if not isinstance(item, dict) or set(item) - allowed:
            problems.append(f"{loc}: must be an object with keys {sorted(allowed)}")
            continue
        fid, any_of = item.get("id"), item.get("any_of")
        if not isinstance(fid, str) or not fid or fid in seen:
            problems.append(f"{loc}: 'id' must be a unique non-empty string")
            continue
        if (
            not isinstance(any_of, list)
            or not any_of
            or not all(isinstance(a, str) and a.strip() for a in any_of)
        ):
            problems.append(f"{loc}: 'any_of' must be a non-empty list of non-empty strings")
            continue
        derived = item.get("derived", False)
        if not isinstance(derived, bool):
            problems.append(f"{loc}: 'derived' must be a boolean")
            continue
        seen.add(fid)
        facts.append(Fact(fact_id=fid, any_of=tuple(any_of), derived=derived))
    return tuple(facts)


def _parse_question(
    obj: Mapping[str, Any], line_no: int, corpus_docs: Mapping[str, str], problems: list[str]
) -> Question | None:
    where = f"questions.jsonl line {line_no}"
    before = len(problems)
    unknown = set(obj) - _QUESTION_KEYS
    if unknown:
        problems.append(f"{where}: unknown keys {sorted(unknown)}")
    qid = obj.get("id")
    if not isinstance(qid, str) or not _ID_RE.match(qid):
        problems.append(f"{where}: 'id' must match {_ID_RE.pattern}")
        qid = f"<line {line_no}>"
    where = f"{where} ({qid})"
    text = obj.get("question")
    if not isinstance(text, str) or not text.strip() or len(text) > 1000:
        problems.append(f"{where}: 'question' must be a non-empty string of at most 1000 chars")
    category = obj.get("category")
    if category not in CATEGORIES:
        problems.append(f"{where}: 'category' must be one of {list(CATEGORIES)}")
    answerable = obj.get("answerable")
    if not isinstance(answerable, bool):
        problems.append(f"{where}: 'answerable' must be true or false")

    spans: list[EvidenceSpan] = []
    raw_evidence = obj.get("evidence", [])
    if not isinstance(raw_evidence, list):
        problems.append(f"{where}: 'evidence' must be a list")
        raw_evidence = []
    for i, ev in enumerate(raw_evidence):
        loc = f"{where} evidence[{i}]"
        if not isinstance(ev, dict) or set(ev) != {"doc", "quote"}:
            problems.append(f"{loc}: must be an object with exactly 'doc' and 'quote'")
            continue
        doc_id, quote = ev["doc"], ev["quote"]
        if not isinstance(doc_id, str) or doc_id not in corpus_docs:
            problems.append(f"{loc}: unknown document {doc_id!r}")
            continue
        if not isinstance(quote, str) or len(quote.strip()) < _MIN_QUOTE_CHARS:
            problems.append(f"{loc}: 'quote' must be at least {_MIN_QUOTE_CHARS} characters")
            continue
        doc_text = corpus_docs[doc_id]
        count = doc_text.count(quote)
        if count != 1:
            problems.append(f"{loc}: quote must occur exactly once in {doc_id}, found {count}")
            continue
        start = doc_text.index(quote)
        spans.append(EvidenceSpan(doc_id=doc_id, quote=quote, start=start, end=start + len(quote)))

    required = _parse_facts(
        obj.get("required_facts"), f"{where} required_facts", problems, allow_derived=True
    )
    stale = _parse_facts(
        obj.get("stale_facts"), f"{where} stale_facts", problems, allow_derived=False
    )
    absent = obj.get("absent_terms", [])
    if not isinstance(absent, list) or not all(isinstance(t, str) and t.strip() for t in absent):
        problems.append(f"{where}: 'absent_terms' must be a list of non-empty strings")
        absent = []
    reference = obj.get("reference_answer", "")
    if not isinstance(reference, str):
        problems.append(f"{where}: 'reference_answer' must be a string")
        reference = ""
    notes = obj.get("notes", "")
    if not isinstance(notes, str):
        problems.append(f"{where}: 'notes' must be a string")
        notes = ""

    if answerable is True:
        if not raw_evidence:
            problems.append(f"{where}: answerable questions need at least one evidence item")
        if not required:
            problems.append(f"{where}: answerable questions need at least one required fact")
        if not reference.strip():
            problems.append(f"{where}: answerable questions need a reference_answer")
        if absent:
            problems.append(f"{where}: 'absent_terms' is only allowed on unanswerable questions")
        quotes = [s.quote for s in spans]
        for fact in required:
            if not fact.derived and not any(
                contains_phrase(q, a) for q in quotes for a in fact.any_of
            ):
                problems.append(
                    f"{where}: required fact {fact.fact_id!r} is not stated in any evidence quote "
                    "(mark it 'derived': true if it is computed from the evidence)"
                )
    elif answerable is False:
        if raw_evidence or required:
            problems.append(
                f"{where}: unanswerable questions must not have evidence or required facts"
            )
        if not absent:
            problems.append(f"{where}: unanswerable questions need 'absent_terms' guarding the gap")
        for term in absent:
            hits = [d for d, t in corpus_docs.items() if contains_phrase(t, term)]
            if hits:
                problems.append(f"{where}: absent term {term!r} occurs in {hits[:3]}")
    for fact in stale:
        if not any(contains_phrase(t, a) for t in corpus_docs.values() for a in fact.any_of):
            problems.append(f"{where}: stale fact {fact.fact_id!r} does not occur in the corpus")

    if len(problems) > before:
        return None
    assert isinstance(text, str) and isinstance(category, str) and isinstance(answerable, bool)
    return Question(
        qid=qid,
        question=text.strip(),
        category=category,
        answerable=answerable,
        evidence=tuple(spans),
        reference_answer=reference.strip(),
        required_facts=required,
        stale_facts=stale,
        absent_terms=tuple(absent),
        notes=notes,
    )


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "dataset.toml"
    if not manifest_path.is_file():
        raise DatasetError(f"missing {manifest_path}")
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise DatasetError(f"invalid TOML in {manifest_path}: {exc}") from exc
    allowed = {"name", "version", "description", "corpus", "questions"}
    unknown = set(manifest) - allowed
    missing = {"name", "version"} - set(manifest)
    if unknown or missing:
        raise DatasetError(
            f"{manifest_path}: unknown keys {sorted(unknown)}, missing {sorted(missing)}"
        )
    return manifest


def load_dataset(path: Path, *, min_chars: int = 1) -> Dataset:
    """Load and fully validate a dataset. Raises ``DatasetError`` listing every problem."""
    root = path.resolve()
    manifest = _read_manifest(root)
    corpus = load_corpus(root / str(manifest.get("corpus", "corpus")), min_chars=min_chars)
    problems = [
        f"corpus issue: {i.source}{f' page {i.page}' if i.page else ''}: {i.kind}: {i.detail}"
        for i in corpus.issues
    ]
    questions_path = root / str(manifest.get("questions", "questions.jsonl"))
    if not questions_path.is_file():
        raise DatasetError(f"missing questions file {questions_path}")
    # Hash newline-normalised text so a CRLF checkout (Git for Windows' default) yields the
    # same questions_sha256 as an LF checkout.
    questions_text = normalize_newlines(questions_path.read_bytes().decode("utf-8-sig"))
    docs = {d.doc_id: d.text for d in corpus.documents}
    questions: list[Question] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(questions_text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"questions.jsonl line {line_no}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(obj, dict):
            problems.append(f"questions.jsonl line {line_no}: must be a JSON object")
            continue
        question = _parse_question(obj, line_no, docs, problems)
        if question is None:
            continue
        if question.qid in seen_ids:
            problems.append(f"questions.jsonl line {line_no}: duplicate id {question.qid!r}")
            continue
        seen_ids.add(question.qid)
        questions.append(question)
    if not questions and not problems:
        problems.append("dataset contains no questions")
    if problems:
        raise DatasetError(f"dataset {root.name} failed validation", problems)
    return Dataset(
        name=str(manifest["name"]),
        version=str(manifest["version"]),
        description=str(manifest.get("description", "")),
        root=str(root),
        corpus=corpus,
        questions=tuple(questions),
        questions_sha256=hashlib.sha256(questions_text.encode("utf-8")).hexdigest(),
    )
