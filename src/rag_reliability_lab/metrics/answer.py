"""Answer metrics. These are deterministic: no LLM judge is involved.

Definitions (see docs/SCORING.md for the full rationale):

* **abstained**: the answer matches an abstention pattern (``parsing.detect_abstention``).
* **fact coverage** (answerable only): share of required facts that appear in the answer.
  A fact appears if any of its aliases occurs as a contiguous token sequence
  (``text.contains_phrase``).
* **stale-only**: the answer states an outdated value (``stale_facts``) and none of the
  current values. Stating both counts as acknowledging the conflict and is allowed.
* **answer_correct**: answerable questions need full fact coverage. Hedging wording doesn't
  matter if every required fact is stated, and a stale-only answer has coverage 0.
  Unanswerable questions need an abstention.
* **answered**: the answer is not a bare abstention, i.e. it either has no abstention
  pattern or it states at least one required fact.
* **citation validity**: every ``[P<n>]`` label refers to a passage the generator received.
* **citation precision**: share of valid cited passages that cover any evidence span.
* **citation evidence recall**: share of evidence spans covered by at least one cited passage.
* **attribution_correct** (answerable and answered): at least one valid citation, no
  invalid citations, and citation evidence recall = 1.
* **supported fact rate**: share of the matched required facts that also appear in the
  text of the cited passages. This is a grounding check at fact level.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..dataset import Fact, Question
from ..models import ScoredChunk
from ..text import contains_token_sequence, match_tokens
from .parsing import detect_abstention, parse_citations
from .retrieval import covers


def fact_present(fact: Fact, tokens: Sequence[str]) -> bool:
    return any(contains_token_sequence(tokens, match_tokens(alias)) for alias in fact.any_of)


@dataclass(frozen=True, slots=True)
class AnswerScores:
    abstained: bool
    abstention_pattern: str | None
    answered: bool
    answer_correct: bool
    fact_coverage: float | None
    facts_matched: tuple[str, ...]
    facts_missing: tuple[str, ...]
    stale_facts_present: tuple[str, ...]
    stale_only: bool
    citations: tuple[int, ...]
    invalid_citations: tuple[int, ...]
    malformed_citations: tuple[str, ...]
    citation_precision: float | None
    citation_evidence_recall: float | None
    attribution_correct: bool | None
    supported_fact_rate: float | None


def score_answer(
    question: Question, answer_text: str, passages: Sequence[ScoredChunk], tau: float
) -> AnswerScores:
    pattern = detect_abstention(answer_text)
    abstained = pattern is not None
    parsed = parse_citations(answer_text)
    valid = tuple(c for c in parsed.labels if 1 <= c <= len(passages))
    invalid = tuple(c for c in parsed.labels if not 1 <= c <= len(passages))
    cited = [passages[c - 1] for c in valid]
    answer_tokens = match_tokens(answer_text)

    stale_present = tuple(f.fact_id for f in question.stale_facts if fact_present(f, answer_tokens))

    if not question.answerable:
        return AnswerScores(
            abstained=abstained,
            abstention_pattern=pattern,
            answered=not abstained,
            answer_correct=abstained,
            fact_coverage=None,
            facts_matched=(),
            facts_missing=(),
            stale_facts_present=stale_present,
            stale_only=False,
            citations=valid,
            invalid_citations=invalid,
            malformed_citations=parsed.malformed,
            citation_precision=None,
            citation_evidence_recall=None,
            attribution_correct=None,
            supported_fact_rate=None,
        )

    matched = tuple(f.fact_id for f in question.required_facts if fact_present(f, answer_tokens))
    missing = tuple(f.fact_id for f in question.required_facts if f.fact_id not in matched)
    coverage_value = len(matched) / len(question.required_facts)
    stale_only = bool(stale_present) and not matched
    answered = (not abstained) or bool(matched)

    citation_precision: float | None = None
    citation_recall: float | None = None
    if cited:
        relevant_cited = [
            p for p in cited if any(covers(p.chunk, s, tau) for s in question.evidence)
        ]
        citation_precision = len(relevant_cited) / len(cited)
        covered = [s for s in question.evidence if any(covers(p.chunk, s, tau) for p in cited)]
        citation_recall = len(covered) / len(question.evidence)
    elif answered:
        citation_recall = 0.0

    attribution: bool | None = None
    if answered:
        attribution = bool(cited) and not invalid and citation_recall == 1.0

    supported_rate: float | None = None
    if matched:
        cited_tokens = match_tokens("\n".join(p.chunk.text for p in cited))
        facts_by_id = {f.fact_id: f for f in question.required_facts}
        supported = [fid for fid in matched if fact_present(facts_by_id[fid], cited_tokens)]
        supported_rate = len(supported) / len(matched)

    return AnswerScores(
        abstained=abstained,
        abstention_pattern=pattern,
        answered=answered,
        answer_correct=coverage_value == 1.0,
        fact_coverage=coverage_value,
        facts_matched=matched,
        facts_missing=missing,
        stale_facts_present=stale_present,
        stale_only=stale_only,
        citations=valid,
        invalid_citations=invalid,
        malformed_citations=parsed.malformed,
        citation_precision=citation_precision,
        citation_evidence_recall=citation_recall,
        attribution_correct=attribution,
        supported_fact_rate=supported_rate,
    )
