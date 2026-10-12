from __future__ import annotations

import pytest

from rag_reliability_lab.dataset import EvidenceSpan, Fact, Question
from rag_reliability_lab.generation import REFUSAL_TEXT
from rag_reliability_lab.metrics.answer import score_answer
from rag_reliability_lab.metrics.retrieval import coverage, score_retrieval
from rag_reliability_lab.models import Chunk, ScoredChunk

SPAN_A = EvidenceSpan("doc.md", "alpha fact", 100, 200)
SPAN_B = EvidenceSpan("other.md", "beta fact", 0, 50)


def _sc(rank: int, doc: str, start: int, end: int, text: str = "") -> ScoredChunk:
    return ScoredChunk(Chunk(f"{doc}::c{rank:04d}", doc, rank, start, end, text), 1.0, rank)


def test_coverage_requires_same_document_and_counts_overlap() -> None:
    assert coverage(_sc(1, "doc.md", 0, 1000).chunk, SPAN_A) == 1.0
    assert coverage(_sc(1, "doc.md", 150, 400).chunk, SPAN_A) == 0.5
    assert coverage(_sc(1, "doc.md", 0, 100).chunk, SPAN_A) == 0.0  # touching, no overlap
    assert coverage(_sc(1, "else.md", 0, 1000).chunk, SPAN_A) == 0.0


def test_retrieval_metrics_single_span() -> None:
    retrieved = [_sc(1, "x.md", 0, 10), _sc(2, "doc.md", 120, 300), _sc(3, "doc.md", 0, 110)]
    s = score_retrieval(retrieved, [SPAN_A], k=3, tau=0.5)
    assert (s.hit, s.complete, s.recall) == (True, True, 1.0)
    assert s.precision == pytest.approx(1 / 3)  # the third chunk covers only 10% of the span
    assert s.reciprocal_rank == 0.5
    s1 = score_retrieval(retrieved, [SPAN_A], k=1, tau=0.5)
    assert (s1.hit, s1.recall, s1.reciprocal_rank, s1.precision) == (False, 0.0, 0.0, 0.0)


def test_retrieval_metrics_multi_span_and_short_lists() -> None:
    retrieved = [_sc(1, "other.md", 0, 60)]  # fewer results than k
    s = score_retrieval(retrieved, [SPAN_A, SPAN_B], k=5, tau=0.5)
    assert s.recall == 0.5 and not s.complete and s.hit
    assert s.precision == pytest.approx(1 / 5)  # denominator is k
    assert s.covered_spans == (1,) and s.missing_spans == (0,)
    empty = score_retrieval([], [SPAN_A], k=3, tau=0.5)
    assert (empty.hit, empty.recall, empty.reciprocal_rank) == (False, 0.0, 0.0)


def test_retrieval_metrics_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        score_retrieval([], [], k=3, tau=0.5)
    with pytest.raises(ValueError):
        score_retrieval([], [SPAN_A], k=0, tau=0.5)


def _question(answerable: bool = True, stale: tuple[Fact, ...] = ()) -> Question:
    return Question(
        qid="q",
        question="What is the rate?",
        category="conflicting" if stale else "factual",
        answerable=answerable,
        evidence=(SPAN_A,) if answerable else (),
        reference_answer="EUR 0.34",
        required_facts=(Fact("rate", ("0.34",)),) if answerable else (),
        stale_facts=stale,
        absent_terms=() if answerable else ("x",),
        notes="",
    )


GOOD = _sc(1, "doc.md", 100, 200, "The rate is EUR 0.34 per km.")
DISTRACTOR = _sc(2, "old.md", 0, 50, "The rate was EUR 0.30 per km.")


def test_correct_answer_with_correct_citation() -> None:
    s = score_answer(_question(), "It is EUR 0.34 per km [P1].", [GOOD, DISTRACTOR], tau=0.5)
    assert s.answer_correct and s.attribution_correct and not s.abstained
    assert s.citation_precision == 1.0 and s.citation_evidence_recall == 1.0
    assert s.supported_fact_rate == 1.0


def test_correct_answer_with_wrong_or_invalid_citation() -> None:
    wrong = score_answer(_question(), "EUR 0.34 [P2].", [GOOD, DISTRACTOR], tau=0.5)
    assert wrong.answer_correct and wrong.attribution_correct is False
    assert wrong.citation_precision == 0.0 and wrong.supported_fact_rate == 0.0
    invalid = score_answer(_question(), "EUR 0.34 [P1][P7].", [GOOD, DISTRACTOR], tau=0.5)
    assert invalid.invalid_citations == (7,) and invalid.attribution_correct is False
    uncited = score_answer(_question(), "EUR 0.34.", [GOOD], tau=0.5)
    assert uncited.attribution_correct is False and uncited.citation_evidence_recall == 0.0


def test_stale_only_vs_mentioning_both() -> None:
    q = _question(stale=(Fact("old", ("0.30",)),))
    stale = score_answer(q, "EUR 0.30 per km [P2].", [GOOD, DISTRACTOR], tau=0.5)
    assert stale.stale_only and not stale.answer_correct and stale.fact_coverage == 0.0
    both = score_answer(q, "Now EUR 0.34 [P1], previously 0.30 [P2].", [GOOD, DISTRACTOR], tau=0.5)
    assert both.answer_correct and not both.stale_only and both.stale_facts_present == ("old",)


def test_abstention_on_answerable_and_hedged_answers() -> None:
    abstain = score_answer(_question(), REFUSAL_TEXT, [GOOD], tau=0.5)
    assert abstain.abstained and not abstain.answered and not abstain.answer_correct
    assert abstain.attribution_correct is None
    hedged = score_answer(
        _question(), "The documents do not state a cap, but the rate is 0.34 [P1].", [GOOD], tau=0.5
    )
    assert hedged.abstained and hedged.answered and hedged.answer_correct


def test_unanswerable_scoring() -> None:
    q = _question(answerable=False)
    assert score_answer(q, REFUSAL_TEXT, [GOOD], tau=0.5).answer_correct
    hallucinated = score_answer(q, "It is 0.34 [P1].", [GOOD], tau=0.5)
    assert not hallucinated.answer_correct and hallucinated.answered
    assert hallucinated.fact_coverage is None and hallucinated.attribution_correct is None
