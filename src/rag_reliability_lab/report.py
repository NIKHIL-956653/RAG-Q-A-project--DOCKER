"""Build the JSON report (the source of truth) and render it as Markdown.

Everything outside the ``environment`` key and the per-question ``latency_ms`` fields is
deterministic: the same code, config and dataset produce byte-identical values.
"""

from __future__ import annotations

import platform
import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from . import __version__
from .dataset import CATEGORIES
from .generation import PROMPT_SHA256, PROMPT_VERSION
from .metrics.retrieval import covers
from .metrics.stats import bootstrap_mean_interval, mean, percentile, wilson_interval
from .runner import DIAGNOSES, EvaluationRun, QuestionResult

SCHEMA_VERSION = "1.0"
_PREVIEW_CHARS = 160


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


def _rate(successes: int, n: int) -> dict[str, Any]:
    ci = wilson_interval(successes, n)
    return {
        "n": n,
        "successes": successes,
        "rate": _r(successes / n) if n else None,
        "ci95": [_r(ci[0]), _r(ci[1])] if ci else None,
        "ci_method": "wilson",
    }


def _mean_stat(values: Sequence[float], resamples: int, seed: int) -> dict[str, Any]:
    ci = bootstrap_mean_interval(values, resamples=resamples, seed=seed)
    return {
        "n": len(values),
        "mean": _r(mean(values)),
        "ci95": [_r(ci[0]), _r(ci[1])] if ci else None,
        "ci_method": "bootstrap_percentile",
    }


def _retrieval_summary(run: EvaluationRun, answerable: list[QuestionResult]) -> dict[str, Any]:
    cfg = run.config
    top_k = cfg.retrieval.top_k
    resamples, seed = cfg.evaluation.bootstrap_resamples, cfg.evaluation.seed
    n = len(answerable)

    def column(k: int, attr: str) -> list[float]:
        # a retrieval error leaves r.retrieval None and counts as 0
        return [float(getattr(r.retrieval[k], attr)) if r.retrieval else 0.0 for r in answerable]

    cutoffs = sorted({*cfg.evaluation.cutoffs, top_k})
    by_cutoff = {
        str(k): {
            "precision": _r(mean(column(k, "precision"))),
            "recall": _r(mean(column(k, "recall"))),
            "mrr": _r(mean(column(k, "reciprocal_rank"))),
            "hit_rate": _r(mean(column(k, "hit"))),
            "complete_rate": _r(mean(column(k, "complete"))),
        }
        for k in cutoffs
    }
    return {
        "scope": "answerable questions",
        "k": top_k,
        "precision_at_k": _mean_stat(column(top_k, "precision"), resamples, seed),
        "recall_at_k": _mean_stat(column(top_k, "recall"), resamples, seed),
        "mrr_at_k": _mean_stat(column(top_k, "reciprocal_rank"), resamples, seed),
        "hit_at_k": _rate(int(sum(column(top_k, "hit"))), n),
        "complete_at_k": _rate(int(sum(column(top_k, "complete"))), n),
        "by_cutoff": by_cutoff,
    }


def _answer_summary(
    run: EvaluationRun, answerable: list[QuestionResult], unanswerable: list[QuestionResult]
) -> dict[str, Any]:
    resamples, seed = run.config.evaluation.bootstrap_resamples, run.config.evaluation.seed
    results = run.results
    answered = [r for r in answerable if r.answer and r.answer.answered]
    with_citations = [
        r.answer.citation_precision
        for r in answerable
        if r.answer and r.answer.citation_precision is not None
    ]
    supported = [
        r.answer.supported_fact_rate
        for r in answerable
        if r.answer and r.answer.supported_fact_rate is not None
    ]
    with_stale = [r for r in results if r.question.stale_facts]
    return {
        "fully_correct": _rate(sum(r.diagnosis == "pass" for r in results), len(results)),
        "answerable": {
            "answer_correct": _rate(
                sum(bool(r.answer and r.answer.answer_correct) for r in answerable), len(answerable)
            ),
            "fact_coverage": _mean_stat(
                [(r.answer.fact_coverage or 0.0) if r.answer else 0.0 for r in answerable],
                resamples,
                seed,
            ),
            "false_abstention": _rate(
                sum(bool(r.answer and not r.answer.answered) for r in answerable), len(answerable)
            ),
            "attribution_correct_when_answered": _rate(
                sum(bool(r.answer and r.answer.attribution_correct) for r in answered),
                len(answered),
            ),
            "citation_precision": {"n": len(with_citations), "mean": _r(mean(with_citations))},
            "supported_fact_rate": {"n": len(supported), "mean": _r(mean(supported))},
            "stale_only_answers": {
                "n": len(with_stale),
                "count": sum(bool(r.answer and r.answer.stale_only) for r in with_stale),
            },
            "stale_value_mentioned": {
                "n": len(with_stale),
                "count": sum(bool(r.answer and r.answer.stale_facts_present) for r in with_stale),
            },
        },
        "unanswerable": {
            "correct_abstention": _rate(
                sum(bool(r.answer and r.answer.abstained) for r in unanswerable), len(unanswerable)
            ),
            "hallucinated_answer": _rate(
                sum(r.diagnosis == "hallucinated_answer" for r in unanswerable), len(unanswerable)
            ),
        },
    }


def _by_category(run: EvaluationRun) -> dict[str, Any]:
    top_k = run.config.retrieval.top_k
    out: dict[str, Any] = {}
    for category in CATEGORIES:
        rows = [r for r in run.results if r.question.category == category]
        if not rows:
            continue
        ans = [r for r in rows if r.question.answerable]
        scored_rows = [r for r in rows if r.diagnosis != "not_applicable"]
        scored = [r.retrieval[top_k] if r.retrieval else None for r in ans]
        out[category] = {
            "n": len(rows),
            "n_answerable": len(ans),
            "hit_at_k": _r(mean([1.0 if s and s.hit else 0.0 for s in scored])),
            "recall_at_k": _r(mean([s.recall if s else 0.0 for s in scored])),
            "answer_correct": _r(
                mean([1.0 if r.answer and r.answer.answer_correct else 0.0 for r in rows])
            )
            if run.generator_info is not None
            else None,
            "pass_rate": _r(mean([1.0 if r.diagnosis == "pass" else 0.0 for r in scored_rows])),
        }
    return out


def _question_entry(r: QuestionResult, top_k: int, tau: float) -> dict[str, Any]:
    q = r.question
    at_k = r.retrieval[top_k] if r.retrieval else None
    retrieved = []
    for sc in r.retrieved:
        in_context = sc.rank <= r.passages_used
        covering = [i for i, span in enumerate(q.evidence) if covers(sc.chunk, span, tau)]
        retrieved.append(
            {
                "rank": sc.rank,
                "chunk_id": sc.chunk.chunk_id,
                "doc_id": sc.chunk.doc_id,
                "start": sc.chunk.start,
                "end": sc.chunk.end,
                "score": _r(sc.score),
                "in_context": in_context,
                "covers_evidence": covering,
                "text": sc.chunk.text if in_context else sc.chunk.text[:_PREVIEW_CHARS],
                "text_truncated": (not in_context) and len(sc.chunk.text) > _PREVIEW_CHARS,
            }
        )
    entry: dict[str, Any] = {
        "id": q.qid,
        "category": q.category,
        "question": q.question,
        "answerable": q.answerable,
        "reference_answer": q.reference_answer or None,
        "evidence": [
            {
                "doc": s.doc_id,
                "quote": s.quote,
                "start": s.start,
                "end": s.end,
                "retrieved_at_k": bool(at_k and i in at_k.covered_spans),
            }
            for i, s in enumerate(q.evidence)
        ],
        "retrieval": None,
        "retrieved": retrieved,
        "answer": None,
        "diagnosis": r.diagnosis,
        "error": r.error,
        "error_stage": r.error_stage,
        "latency_ms": {k: _r(v) for k, v in r.latency_ms.items()},
    }
    if r.retrieval and at_k:
        entry["retrieval"] = {
            "k": top_k,
            "precision": _r(at_k.precision),
            "recall": _r(at_k.recall),
            "hit": at_k.hit,
            "complete": at_k.complete,
            "reciprocal_rank": _r(at_k.reciprocal_rank),
            "by_cutoff": {
                str(k): {
                    "recall": _r(s.recall),
                    "hit": s.hit,
                    "reciprocal_rank": _r(s.reciprocal_rank),
                }
                for k, s in sorted(r.retrieval.items())
            },
        }
    if r.answer is not None:
        a = r.answer
        entry["answer"] = {
            "text": r.answer_text,
            "abstained": a.abstained,
            "abstention_pattern": a.abstention_pattern,
            "answered": a.answered,
            "answer_correct": a.answer_correct,
            "fact_coverage": _r(a.fact_coverage),
            "facts_matched": list(a.facts_matched),
            "facts_missing": list(a.facts_missing),
            "stale_facts_present": list(a.stale_facts_present),
            "stale_only": a.stale_only,
            "citations": list(a.citations),
            "invalid_citations": list(a.invalid_citations),
            "malformed_citations": list(a.malformed_citations),
            "citation_precision": _r(a.citation_precision),
            "citation_evidence_recall": _r(a.citation_evidence_recall),
            "attribution_correct": a.attribution_correct,
            "supported_fact_rate": _r(a.supported_fact_rate),
        }
    return entry


def _latency(results: Sequence[QuestionResult], stage: str) -> dict[str, float | None] | None:
    values = [r.latency_ms[stage] for r in results if stage in r.latency_ms]
    if not values:
        return None
    return {
        "p50": _r(percentile(values, 50)),
        "p95": _r(percentile(values, 95)),
        "max": _r(max(values)),
    }


def build_report(
    run: EvaluationRun, *, git_commit: str | None = None, config_file: str | None = None
) -> dict[str, Any]:
    cfg = run.config
    results = list(run.results)
    answerable = [r for r in results if r.question.answerable]
    unanswerable = [r for r in results if not r.question.answerable]
    chunk_lengths = [len(c.text) for c in run.chunks]
    diag = Counter(r.diagnosis for r in results)
    try:
        import pypdf

        pypdf_version: str | None = pypdf.__version__
    except ImportError:  # pragma: no cover
        pypdf_version = None

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "name": cfg.name,
            "description": cfg.description,
            "package_version": __version__,
            "config": cfg.to_dict(),
            "config_sha256": cfg.sha256(),
            "dataset": {
                "name": run.dataset.name,
                "version": run.dataset.version,
                "questions_sha256": run.dataset.questions_sha256,
                "corpus_fingerprint": run.dataset.corpus.fingerprint,
                "n_documents": len(run.dataset.corpus.documents),
                "n_questions_total": len(run.dataset.questions),
                "n_questions_evaluated": len(results),
            },
            "components": {
                "chunker": {
                    "strategy": cfg.chunking.strategy,
                    "chunk_size": cfg.chunking.chunk_size,
                    "chunk_overlap": cfg.chunking.chunk_overlap,
                    "n_chunks": len(run.chunks),
                },
                "retriever": run.retriever_info,
                "generator": run.generator_info,
                "prompt": {"version": PROMPT_VERSION, "sha256": PROMPT_SHA256},
            },
            "scoring": {
                "evidence_overlap": cfg.evaluation.evidence_overlap,
                "cutoffs": list(cfg.evaluation.cutoffs),
                "bootstrap_resamples": cfg.evaluation.bootstrap_resamples,
                "seed": cfg.evaluation.seed,
                "errors_count_as_failures": True,
            },
        },
        "ingestion": {
            "files": list(run.dataset.corpus.files),
            "n_documents": len(run.dataset.corpus.documents),
            "n_chunks": len(run.chunks),
            "chunk_chars": {
                "min": min(chunk_lengths) if chunk_lengths else None,
                "median": statistics.median(chunk_lengths) if chunk_lengths else None,
                "max": max(chunk_lengths) if chunk_lengths else None,
            },
            "issues": [
                {"source": i.source, "page": i.page, "kind": i.kind, "detail": i.detail}
                for i in run.dataset.corpus.issues
            ],
        },
        "summary": {
            "n_questions": len(results),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "n_errors": sum(r.error is not None for r in results),
            "retrieval": _retrieval_summary(run, answerable) if answerable else None,
            "answer": _answer_summary(run, answerable, unanswerable)
            if run.generator_info
            else None,
            "diagnosis_counts": {d: diag.get(d, 0) for d in DIAGNOSES},
            "by_category": _by_category(run),
        },
        "questions": [
            _question_entry(r, cfg.retrieval.top_k, cfg.evaluation.evidence_overlap)
            for r in results
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(terse=True),
            "pypdf": pypdf_version,
            "git_commit": git_commit,
            "config_file": config_file,
            "started_at": run.started_at,
            "duration_s": _r(run.duration_s),
            "index_ms": _r(run.index_ms),
            "latency_ms": {
                "retrieval": _latency(results, "retrieval"),
                "generation": _latency(results, "generation"),
            },
        },
    }


# --------------------------------------------------------------------------------------------
# Markdown rendering (from the JSON dict, so old reports can be re-rendered)
# --------------------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(stat: dict[str, Any] | None, fmt: str = "pct") -> str:
    if not stat or not stat.get("ci95"):
        return "n/a"
    lo, hi = stat["ci95"]
    return f"[{_pct(lo)}, {_pct(hi)}]" if fmt == "pct" else f"[{_num(lo)}, {_num(hi)}]"


def _rate_row(label: str, stat: dict[str, Any]) -> str:
    return f"| {label} | {_pct(stat['rate'])} ({stat['successes']}/{stat['n']}) | {_ci(stat)} |"


def _mean_row(label: str, stat: dict[str, Any]) -> str:
    return f"| {label} | {_num(stat['mean'])} (n={stat['n']}) | {_ci(stat, 'num')} |"


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    run, summary = report["run"], report["summary"]
    ds, comp = run["dataset"], run["components"]
    gen = comp["generator"]
    lines: list[str] = [
        f"# RAG evaluation report: {run['name']}",
        "",
        run["description"] or "",
        "",
        f"- **Dataset:** {ds['name']} v{ds['version']}. {ds['n_questions_evaluated']} of "
        f"{ds['n_questions_total']} questions evaluated ({summary['n_answerable']} answerable, "
        f"{summary['n_unanswerable']} unanswerable), {ds['n_documents']} documents",
        f"- **Chunking:** {comp['chunker']['strategy']} {comp['chunker']['chunk_size']}/"
        f"{comp['chunker']['chunk_overlap']} -> {comp['chunker']['n_chunks']} chunks",
        f"- **Retriever:** {', '.join(f'{k}={v}' for k, v in comp['retriever'].items())}; "
        f"top_k={run['config']['retrieval']['top_k']}",
        "- **Generator:** "
        + (", ".join(f"{k}={v}" for k, v in gen.items()) if gen else "none (retrieval-only run)"),
        f"- **Config sha256:** `{run['config_sha256'][:16]}` · "
        f"**package** {run['package_version']}",
        f"- **Errors:** {summary['n_errors']} (errors count as failures)",
        "",
        "> **Read this first.** This is a small, hand-built dataset. Every rate comes with a 95% "
        "confidence interval. Differences smaller than the interval width are not evidence "
        "that one configuration is better. Retrieval and answer quality are scored separately "
        "and deterministically (no LLM judge). See `docs/SCORING.md`.",
        "",
    ]

    ret = summary.get("retrieval")
    if ret:
        k = ret["k"]
        lines += [
            f"## Retrieval quality (answerable questions, k={k})",
            "",
            "| Metric | Value | 95% CI |",
            "|---|---|---|",
            _rate_row(f"Hit@{k} (any evidence retrieved)", ret["hit_at_k"]),
            _rate_row(f"Complete@{k} (all evidence retrieved)", ret["complete_at_k"]),
            _mean_row(f"Recall@{k}", ret["recall_at_k"]),
            _mean_row(f"Precision@{k}", ret["precision_at_k"]),
            _mean_row(f"MRR@{k}", ret["mrr_at_k"]),
            "",
            "| Cutoff | Hit | Complete | Recall | Precision | MRR |",
            "|---|---|---|---|---|---|",
        ]
        for cutoff, row in ret["by_cutoff"].items():
            lines.append(
                f"| @{cutoff} | {_pct(row['hit_rate'])} | {_pct(row['complete_rate'])} | "
                f"{_num(row['recall'])} | {_num(row['precision'])} | {_num(row['mrr'])} |"
            )
        lines.append("")

    ans = summary.get("answer")
    if ans:
        a, u = ans["answerable"], ans["unanswerable"]
        lines += [
            "## Answer quality",
            "",
            "| Metric | Value | 95% CI |",
            "|---|---|---|",
            _rate_row(
                "**Fully correct** (answer + attribution, or correct abstention)",
                ans["fully_correct"],
            ),
            _rate_row("Answerable: answer correct (all required facts)", a["answer_correct"]),
            _mean_row("Answerable: fact coverage", a["fact_coverage"]),
            _rate_row("Answerable: false abstention", a["false_abstention"]),
            _rate_row(
                "Answerable: attribution correct (when answered)",
                a["attribution_correct_when_answered"],
            ),
            _rate_row("Unanswerable: correct abstention", u["correct_abstention"]),
            _rate_row("Unanswerable: hallucinated answer", u["hallucinated_answer"]),
            "",
            f"Citation precision (answers with citations): {_num(a['citation_precision']['mean'])} "
            f"(n={a['citation_precision']['n']}) · Supported-fact rate: "
            f"{_num(a['supported_fact_rate']['mean'])} (n={a['supported_fact_rate']['n']}) · "
            f"Stale-only answers: {a['stale_only_answers']['count']}/"
            f"{a['stale_only_answers']['n']} · "
            f"Stale value mentioned (alongside or instead of the current one): "
            f"{a['stale_value_mentioned']['count']}/{a['stale_value_mentioned']['n']}",
            "",
        ]

    lines += ["## Diagnosis", "", "| Diagnosis | Count |", "|---|---|"]
    lines += [f"| {d} | {c} |" for d, c in summary["diagnosis_counts"].items() if c]
    lines += [
        "",
        "## By category",
        "",
        "| Category | n | Hit@k | Recall@k | Answer correct | Pass |",
        "|---|---|---|---|---|---|",
    ]
    for cat, row in summary["by_category"].items():
        lines.append(
            f"| {cat} | {row['n']} | {_pct(row['hit_at_k'])} | {_num(row['recall_at_k'])} | "
            f"{_pct(row['answer_correct'])} | {_pct(row['pass_rate'])} |"
        )
    lines += ["", "Per-category n is small; treat these rows as diagnostic, not as estimates.", ""]

    failures = [q for q in report["questions"] if q["diagnosis"] != "pass"]
    lines += [f"## Failed questions ({len(failures)})", ""]
    for q in failures:
        lines += [
            f"### {q['id']} · {q['category']} · `{q['diagnosis']}`",
            "",
            f"**Q:** {q['question']}",
            "",
        ]
        if q["error"]:
            lines += [f"**Error ({q['error_stage']}):** `{_escape(q['error'])}`", ""]
        if q["evidence"]:
            lines.append("**Expected evidence:**")
            for ev in q["evidence"]:
                mark = "retrieved" if ev["retrieved_at_k"] else "**MISSING**"
                lines.append(f'- [{mark}] `{ev["doc"]}`: "{_escape(ev["quote"])}"')
            lines.append("")
        else:
            lines += ["**Expected:** abstention (not answerable from the corpus).", ""]
        in_context = [p for p in q["retrieved"] if p["in_context"]]
        if in_context:
            lines += ["| Rank | Chunk | Score | Covers evidence |", "|---|---|---|---|"]
            for p in in_context:
                covers = ", ".join(str(i) for i in p["covers_evidence"]) or "-"
                lines.append(f"| {p['rank']} | `{p['chunk_id']}` | {_num(p['score'])} | {covers} |")
            lines.append("")
        else:
            lines += ["_No passages retrieved._", ""]
        a = q["answer"]
        if a:
            lines += [f"**Answer:** {_escape(a['text'] or '')}", ""]
            details = []
            if a["facts_missing"]:
                details.append(f"missing facts: {', '.join(a['facts_missing'])}")
            if a["stale_facts_present"]:
                details.append(f"stale facts stated: {', '.join(a['stale_facts_present'])}")
            if a["citations"] or a["invalid_citations"]:
                details.append(f"citations: {a['citations']} invalid: {a['invalid_citations']}")
            if a["abstention_pattern"]:
                details.append(f"abstention: {a['abstention_pattern']}")
            if details:
                lines += [f"_{'; '.join(details)}_", ""]
        if q["reference_answer"]:
            lines += [f"**Reference:** {q['reference_answer']}", ""]

    ing = report["ingestion"]
    lines += [
        "## Ingestion",
        "",
        f"{len(ing['files'])} files -> {ing['n_documents']} documents -> {ing['n_chunks']} chunks "
        f"(chunk chars min/median/max: {ing['chunk_chars']['min']}/{ing['chunk_chars']['median']}/"
        f"{ing['chunk_chars']['max']}). Issues: {len(ing['issues'])}.",
        "",
    ]
    for issue in ing["issues"]:
        page = f" page {issue['page']}" if issue["page"] else ""
        lines.append(f"- `{issue['source']}`{page}: {issue['kind']}: {issue['detail']}")

    env = report["environment"]
    lat = env["latency_ms"]

    def lat_text(stage: str) -> str:
        v = lat.get(stage)
        return "n/a" if not v else f"p50 {v['p50']:.2f} ms, p95 {v['p95']:.2f} ms"

    lines += [
        "",
        "## Latency and environment (not deterministic)",
        "",
        f"Retrieval {lat_text('retrieval')} · Generation {lat_text('generation')} · "
        f"index build {env['index_ms']:.1f} ms · total {env['duration_s']:.2f} s · Python "
        f"{env['python']} on {env['platform']} · pypdf {env['pypdf']} · "
        f"git `{env['git_commit'] or 'n/a'}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        "pip install -e .",
        f"rag-lab eval --config {env.get('config_file') or '<config file>'} --out <dir>",
        "```",
        "",
        f"Expected hashes: config `{run['config_sha256']}`, questions `{ds['questions_sha256']}`, "
        f"corpus `{ds['corpus_fingerprint']}`.",
        "",
    ]
    return "\n".join(lines)
