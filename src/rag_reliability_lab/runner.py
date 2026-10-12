"""Evaluation orchestration: dataset -> chunks -> index -> per-question retrieve/generate/score.

Every question is evaluated in isolation, so a failure in one question never aborts the run.
Errors count as failures in all rates, and the report lists them. Nothing is silently dropped.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .chunking import RecursiveCharacterSplitter, chunk_documents
from .config import RunConfig
from .dataset import Dataset, Question, load_dataset
from .errors import ConfigError
from .generation import Generator
from .metrics.answer import AnswerScores, score_answer
from .metrics.retrieval import RetrievalScores, score_retrieval
from .models import Chunk, ScoredChunk
from .retrieval import Retriever

DIAGNOSES: tuple[str, ...] = (
    "pass",
    "retrieval_miss",
    "false_abstention",
    "stale_answer",
    "generation_miss",
    "citation_error",
    "hallucinated_answer",
    "error",
    "not_applicable",  # unanswerable question in a retrieval-only run
)


def build_retriever(config: RunConfig) -> Retriever:
    r = config.retrieval
    if r.method == "bm25":
        from .retrieval.bm25 import BM25Retriever

        return BM25Retriever(k1=r.bm25.k1, b=r.bm25.b)
    if r.method == "dense":
        from .retrieval.dense import DenseRetriever, SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(
            r.dense.model,
            device=r.dense.device,
            batch_size=r.dense.batch_size,
            normalize_embeddings=r.dense.normalize_embeddings,
            replace_newlines=r.dense.replace_newlines,
        )
        return DenseRetriever(embedder, metric=r.dense.metric)
    raise ConfigError(f"unknown retrieval method {r.method!r}")  # pragma: no cover


def build_generator(config: RunConfig) -> Generator | None:
    g = config.generation
    if g.provider == "none":
        return None
    if g.provider == "extractive":
        from .generation.extractive import ExtractiveGenerator

        return ExtractiveGenerator(
            max_sentences=g.extractive.max_sentences, min_overlap=g.extractive.min_overlap
        )
    if g.provider == "gemini":
        from .generation.gemini import GeminiGenerator

        c = g.gemini
        return GeminiGenerator(
            c.model,
            temperature=c.temperature,
            max_output_tokens=c.max_output_tokens,
            timeout_s=c.timeout_s,
            max_retries=c.max_retries,
            backoff_s=c.backoff_s,
            api_key_env=c.api_key_env,
        )
    raise ConfigError(f"unknown generation provider {g.provider!r}")  # pragma: no cover


@dataclass(frozen=True)
class QuestionResult:
    question: Question
    retrieved: tuple[ScoredChunk, ...]
    passages_used: int
    retrieval: dict[int, RetrievalScores] | None  # by cutoff; None if unanswerable
    answer_text: str | None
    answer: AnswerScores | None
    diagnosis: str
    error: str | None = None
    error_stage: str | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRun:
    config: RunConfig
    dataset: Dataset
    chunks: tuple[Chunk, ...]
    retriever_info: dict[str, object]
    generator_info: dict[str, object] | None
    results: tuple[QuestionResult, ...]
    started_at: str
    duration_s: float
    index_ms: float


def diagnose(
    question: Question,
    at_top_k: RetrievalScores | None,
    answer: AnswerScores | None,
    error: str | None,
) -> str:
    """Name the first pipeline stage that failed (see docs/SCORING.md)."""
    if error is not None:
        return "error"
    if answer is None:  # retrieval-only run
        if at_top_k is None:
            return "not_applicable"  # nothing to score without an answer
        return "pass" if at_top_k.complete else "retrieval_miss"
    if not question.answerable:
        return "pass" if answer.abstained else "hallucinated_answer"
    if answer.answer_correct:
        return "pass" if answer.attribution_correct else "citation_error"
    if at_top_k is not None and not at_top_k.complete:
        return "retrieval_miss"
    if answer.stale_only:
        return "stale_answer"
    if not answer.answered:
        return "false_abstention"
    return "generation_miss"


def _evaluate_question(
    question: Question,
    retriever: Retriever,
    generator: Generator | None,
    config: RunConfig,
    clock: Callable[[], float],
) -> QuestionResult:
    top_k = config.retrieval.top_k
    depth = max(top_k, *config.evaluation.cutoffs)
    tau = config.evaluation.evidence_overlap
    latency: dict[str, float] = {}

    t0 = clock()
    try:
        retrieved = tuple(retriever.retrieve(question.question, depth))
    except Exception as exc:  # noqa: BLE001 - recorded per question
        latency["retrieval"] = (clock() - t0) * 1000
        return QuestionResult(
            question,
            (),
            0,
            None,
            None,
            None,
            "error",
            error=f"{type(exc).__name__}: {exc}",
            error_stage="retrieval",
            latency_ms=latency,
        )
    latency["retrieval"] = (clock() - t0) * 1000

    by_cutoff: dict[int, RetrievalScores] | None = None
    if question.answerable:
        cutoffs = sorted({*config.evaluation.cutoffs, top_k})
        by_cutoff = {k: score_retrieval(retrieved, question.evidence, k, tau) for k in cutoffs}
    at_top_k = by_cutoff[top_k] if by_cutoff else None

    passages: Sequence[ScoredChunk] = retrieved[:top_k]
    if generator is None:
        diagnosis = diagnose(question, at_top_k, None, None)
        return QuestionResult(
            question, retrieved, len(passages), by_cutoff, None, None, diagnosis, latency_ms=latency
        )

    t1 = clock()
    try:
        output = generator.generate(question.question, passages)
    except Exception as exc:  # noqa: BLE001 - recorded per question
        latency["generation"] = (clock() - t1) * 1000
        return QuestionResult(
            question,
            retrieved,
            len(passages),
            by_cutoff,
            None,
            None,
            "error",
            error=f"{type(exc).__name__}: {exc}",
            error_stage="generation",
            latency_ms=latency,
        )
    latency["generation"] = (clock() - t1) * 1000
    answer = score_answer(question, output.text, passages, tau)
    diagnosis = diagnose(question, at_top_k, answer, None)
    return QuestionResult(
        question,
        retrieved,
        len(passages),
        by_cutoff,
        output.text,
        answer,
        diagnosis,
        latency_ms=latency,
    )


def run_evaluation(
    config: RunConfig,
    dataset_dir: Path,
    *,
    question_ids: Sequence[str] | None = None,
    limit: int | None = None,
    retriever: Retriever | None = None,
    generator: Generator | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EvaluationRun:
    """Run a full evaluation. ``retriever`` and ``generator`` can be injected (tests, custom
    pipelines); otherwise they are built from ``config``."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    t_start = clock()
    dataset = load_dataset(dataset_dir, min_chars=config.min_chars)

    questions = list(dataset.questions)
    if question_ids:
        wanted = set(question_ids)
        unknown = wanted - {q.qid for q in questions}
        if unknown:
            raise ConfigError(f"unknown question id(s): {sorted(unknown)}")
        questions = [q for q in questions if q.qid in wanted]
    if limit is not None:
        if limit < 1:
            raise ConfigError("limit must be >= 1")
        questions = questions[:limit]

    splitter = RecursiveCharacterSplitter(config.chunking.chunk_size, config.chunking.chunk_overlap)
    chunks = tuple(chunk_documents(dataset.corpus.documents, splitter))

    retriever = retriever if retriever is not None else build_retriever(config)
    if generator is None and config.generation.provider != "none":
        generator = build_generator(config)
    t_index = clock()
    retriever.index(chunks)
    index_ms = (clock() - t_index) * 1000

    results = tuple(_evaluate_question(q, retriever, generator, config, clock) for q in questions)
    return EvaluationRun(
        config=config,
        dataset=dataset,
        chunks=chunks,
        retriever_info=retriever.describe(),
        generator_info=generator.describe() if generator is not None else None,
        results=results,
        started_at=started,
        duration_s=clock() - t_start,
        index_ms=index_ms,
    )
