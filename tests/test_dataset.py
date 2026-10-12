from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rag_reliability_lab.dataset import CATEGORIES, load_dataset
from rag_reliability_lab.errors import DatasetError

from .conftest import DATASET_DIR, TINY_DOCS, TINY_QUESTIONS


def test_shipped_dataset_is_valid_and_balanced() -> None:
    ds = load_dataset(DATASET_DIR)
    assert ds.name == "tarnwick-v1"
    assert len(ds.questions) == 56
    assert {q.category for q in ds.questions} == set(CATEGORIES)
    assert sum(not q.answerable for q in ds.questions) == 9
    for q in ds.questions:
        docs = ds.corpus.document_map()
        for span in q.evidence:
            assert docs[span.doc_id].text[span.start : span.end] == span.quote


def test_tiny_dataset_loads(write_dataset: Callable[..., Path]) -> None:
    ds = load_dataset(write_dataset(TINY_DOCS, TINY_QUESTIONS))
    assert [q.qid for q in ds.questions] == ["q1", "q2"]
    assert ds.questions[0].evidence[0].start == TINY_DOCS["alpha.md"].index("The alpha widget")


def _broken(mutate: Callable[[list[dict[str, Any]]], None]) -> list[dict[str, Any]]:
    questions = copy.deepcopy(TINY_QUESTIONS)
    mutate(questions)
    return questions


BROKEN_CASES: dict[str, Callable[[list[dict[str, Any]]], None]] = {
    "duplicate id": lambda qs: qs[1].update(id="q1"),
    "unknown document": lambda qs: qs[0]["evidence"][0].update(doc="missing.md"),
    "quote not in document": lambda qs: qs[0]["evidence"][0].update(
        quote="The alpha widget is huge."
    ),
    "fact not in evidence": lambda qs: qs[0]["required_facts"][0].update(any_of=["99 kg"]),
    "unknown key": lambda qs: qs[0].update(colour="red"),
    "bad category": lambda qs: qs[0].update(category="trivia"),
    "unanswerable with evidence": lambda qs: qs[1].update(evidence=qs[0]["evidence"]),
    "absent term present in corpus": lambda qs: qs[1].update(absent_terms=["beta widget"]),
    "unanswerable without guard": lambda qs: qs[1].pop("absent_terms"),
    "answerable without facts": lambda qs: qs[0].update(required_facts=[]),
    "stale fact not in corpus": lambda qs: qs[0].update(
        stale_facts=[{"id": "x", "any_of": ["77 kg"]}]
    ),
    "bad id": lambda qs: qs[0].update(id="Q 1!"),
}


@pytest.mark.parametrize("case", sorted(BROKEN_CASES))
def test_validator_rejects(case: str, write_dataset: Callable[..., Path]) -> None:
    root = write_dataset(TINY_DOCS, _broken(BROKEN_CASES[case]))
    with pytest.raises(DatasetError) as info:
        load_dataset(root)
    assert info.value.problems, case


def test_duplicate_quote_is_ambiguous(write_dataset: Callable[..., Path]) -> None:
    docs = {"alpha.md": "The alpha widget weighs 12 kg. The alpha widget weighs 12 kg."}
    with pytest.raises(DatasetError, match="exactly once"):
        load_dataset(write_dataset(docs, TINY_QUESTIONS[:1]))


def test_invalid_json_line_reported(write_dataset: Callable[..., Path]) -> None:
    root = write_dataset(TINY_DOCS, TINY_QUESTIONS)
    with (root / "questions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_dataset(root)


def test_corpus_issues_fail_validation(write_dataset: Callable[..., Path]) -> None:
    root = write_dataset({**TINY_DOCS, "empty.md": "  "}, TINY_QUESTIONS)
    with pytest.raises(DatasetError, match="empty_text"):
        load_dataset(root)


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"dataset\.toml"):
        load_dataset(tmp_path)


def test_questions_hash_ignores_line_endings(write_dataset: Callable[..., Path]) -> None:
    """Git for Windows checks files out with CRLF by default; hashes must not change."""
    root = write_dataset(TINY_DOCS, TINY_QUESTIONS)
    lf_hash = load_dataset(root).questions_sha256
    path = root / "questions.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert load_dataset(root).questions_sha256 == lf_hash
