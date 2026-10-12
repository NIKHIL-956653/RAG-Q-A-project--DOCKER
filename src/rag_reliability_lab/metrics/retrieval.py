"""Retrieval metrics against span-level evidence.

A retrieved chunk **covers** an evidence span when it comes from the same document and
contains at least ``tau`` of the span's characters (default 0.5). A chunk is **relevant**
when it covers at least one evidence span of the question.

For a cutoff k, with E = the question's evidence spans and R_k = the top-k chunks:

* precision@k = |relevant chunks in R_k| / k          (k is the denominator even if fewer
                                                        chunks were returned)
* recall@k    = |spans in E covered by R_k| / |E|
* hit@k       = 1 if any chunk in R_k is relevant, else 0
* complete@k  = 1 if every span in E is covered by R_k, else 0
* rr@k        = 1 / rank of the first relevant chunk in R_k, else 0  (its mean is MRR@k)

Unanswerable questions have no evidence. They are excluded from retrieval averages and
scored on answer behaviour only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..dataset import EvidenceSpan
from ..models import Chunk, ScoredChunk


def coverage(chunk: Chunk, span: EvidenceSpan) -> float:
    """Share of the evidence span's characters that lie inside the chunk (0 to 1)."""
    if chunk.doc_id != span.doc_id or span.end <= span.start:
        return 0.0
    overlap = min(chunk.end, span.end) - max(chunk.start, span.start)
    return max(0, overlap) / (span.end - span.start)


def covers(chunk: Chunk, span: EvidenceSpan, tau: float) -> bool:
    return coverage(chunk, span) >= tau


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    k: int
    precision: float
    recall: float
    hit: bool
    complete: bool
    reciprocal_rank: float
    covered_spans: tuple[int, ...]  # indices into the question's evidence
    missing_spans: tuple[int, ...]


def score_retrieval(
    retrieved: Sequence[ScoredChunk], evidence: Sequence[EvidenceSpan], k: int, tau: float
) -> RetrievalScores:
    if k < 1:
        raise ValueError("k must be >= 1")
    if not evidence:
        raise ValueError("retrieval metrics are undefined for questions without evidence")
    top = list(retrieved[:k])
    relevant = [any(covers(sc.chunk, span, tau) for span in evidence) for sc in top]
    covered = [
        i for i, span in enumerate(evidence) if any(covers(sc.chunk, span, tau) for sc in top)
    ]
    missing = [i for i in range(len(evidence)) if i not in covered]
    first = next((pos for pos, flag in enumerate(relevant, start=1) if flag), None)
    return RetrievalScores(
        k=k,
        precision=sum(relevant) / k,
        recall=len(covered) / len(evidence),
        hit=any(relevant),
        complete=not missing,
        reciprocal_rank=(1.0 / first) if first else 0.0,
        covered_spans=tuple(covered),
        missing_spans=tuple(missing),
    )
