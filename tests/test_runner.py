from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from rag_reliability_lab.config import parse_config
from rag_reliability_lab.errors import ConfigError
from rag_reliability_lab.generation import GeneratorOutput
from rag_reliability_lab.models import Chunk, ScoredChunk
from rag_reliability_lab.report import build_report
from rag_reliability_lab.retrieval.bm25 import BM25Retriever
from rag_reliability_lab.runner import diagnose, run_evaluation

from .conftest import TINY_DOCS, TINY_QUESTIONS


def _config(**overrides: object) -> object:
    return parse_config(
        {"name": "tiny", "dataset": ".", "evaluation": {"cutoffs": [1, 3]}, **overrides}
    )


def test_end_to_end_on_tiny_dataset(write_dataset: Callable[..., Path]) -> None:
    root = write_dataset(TINY_DOCS, TINY_QUESTIONS)
    run = run_evaluation(_config(), root)  # type: ignore[arg-type]
    by_id = {r.question.qid: r for r in run.results}
    assert by_id["q1"].diagnosis == "pass"
    assert by_id["q1"].answer is not None and by_id["q1"].answer.answer_correct
    assert by_id["q2"].diagnosis == "pass"  # abstained on the unanswerable question
    assert by_id["q2"].retrieval is None


class ExplodingRetriever(BM25Retriever):
    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        if "alpha" in query:
            raise RuntimeError("index offline")
        return super().retrieve(query, k)


class FlakyGenerator:
    name = "flaky"

    def generate(self, question: str, passages: Sequence[ScoredChunk]) -> GeneratorOutput:
        raise TimeoutError("model timed out")

    def describe(self) -> dict[str, object]:
        return {"provider": "flaky"}


def test_errors_are_isolated_per_question_and_count_as_failures(
    write_dataset: Callable[..., Path],
) -> None:
    root = write_dataset(TINY_DOCS, TINY_QUESTIONS)
    run = run_evaluation(_config(), root, retriever=ExplodingRetriever())  # type: ignore[arg-type]
    q1, q2 = run.results
    assert (q1.diagnosis, q1.error_stage) == ("error", "retrieval")
    assert "index offline" in (q1.error or "")
    assert q2.diagnosis == "pass"
    report = build_report(run)
    assert report["summary"]["n_errors"] == 1
    assert report["summary"]["retrieval"]["hit_at_k"]["successes"] == 0  # error counted as miss

    run = run_evaluation(_config(), root, generator=FlakyGenerator())  # type: ignore[arg-type]
    assert [(r.diagnosis, r.error_stage) for r in run.results] == [
        ("error", "generation"),
        ("error", "generation"),
    ]
    assert build_report(run)["summary"]["answer"]["fully_correct"]["successes"] == 0


def test_retrieval_only_mode(write_dataset: Callable[..., Path]) -> None:
    root = write_dataset(TINY_DOCS, TINY_QUESTIONS)
    run = run_evaluation(_config(generation={"provider": "none"}), root)  # type: ignore[arg-type]
    assert run.generator_info is None
    assert all(r.answer is None for r in run.results)
    assert [r.diagnosis for r in run.results] == ["pass", "not_applicable"]
    report = build_report(run)
    assert report["summary"]["answer"] is None
    assert report["summary"]["by_category"]["unanswerable"]["pass_rate"] is None


def test_question_filter_and_limit(write_dataset: Callable[..., Path]) -> None:
    root = write_dataset(TINY_DOCS, TINY_QUESTIONS)
    cfg = _config()
    assert [r.question.qid for r in run_evaluation(cfg, root, question_ids=["q2"]).results] == [
        "q2"
    ]  # type: ignore[arg-type]
    assert len(run_evaluation(cfg, root, limit=1).results) == 1  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="unknown question"):
        run_evaluation(cfg, root, question_ids=["nope"])  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        run_evaluation(cfg, root, limit=0)  # type: ignore[arg-type]


def test_diagnose_branches() -> None:
    from rag_reliability_lab.dataset import EvidenceSpan, Fact, Question
    from rag_reliability_lab.metrics.answer import score_answer
    from rag_reliability_lab.metrics.retrieval import score_retrieval

    span = EvidenceSpan("d.md", "x" * 20, 0, 20)
    q = Question("q", "rate?", "conflicting", True, (span,), "1", (Fact("f", ("0.34",)),),
                 (Fact("s", ("0.30",)),), (), "")  # fmt: skip
    hit = [ScoredChunk(Chunk("d.md::c0000", "d.md", 0, 0, 20, "rate 0.34"), 1.0, 1)]
    miss = [ScoredChunk(Chunk("e.md::c0000", "e.md", 0, 0, 20, "other"), 1.0, 1)]
    r_hit, r_miss = score_retrieval(hit, [span], 1, 0.5), score_retrieval(miss, [span], 1, 0.5)

    def d(answer: str, passages: list[ScoredChunk], at_k: object) -> str:
        return diagnose(q, at_k, score_answer(q, answer, passages, 0.5), None)  # type: ignore[arg-type]

    assert d("0.34 [P1]", hit, r_hit) == "pass"
    assert d("0.34", hit, r_hit) == "citation_error"
    assert d("unknown", miss, r_miss) == "retrieval_miss"
    assert d("0.30 [P1]", hit, r_hit) == "stale_answer"
    assert (
        d("I could not find the answer in the provided documents.", hit, r_hit)
        == "false_abstention"
    )
    assert d("something else [P1]", hit, r_hit) == "generation_miss"
    assert diagnose(q, r_hit, None, "boom") == "error"
    unanswerable = Question("u", "?", "unanswerable", False, (), "", (), (), ("z",), "")
    assert diagnose(unanswerable, None, score_answer(unanswerable, "It is 5.", [], 0.5), None) == (
        "hallucinated_answer"
    )
