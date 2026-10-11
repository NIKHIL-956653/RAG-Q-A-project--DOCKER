"""Command-line interface: ``rag-lab``.

Exit codes: 0 success; 1 invalid input (config, dataset or arguments); 2 the run completed but
at least one question raised an error (the report is still written).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import load_config, resolve_dataset_dir
from .dataset import load_dataset
from .errors import LabError
from .report import build_report, render_markdown
from .runner import run_evaluation

EXIT_OK, EXIT_INVALID, EXIT_QUESTION_ERRORS = 0, 1, 2


def _git_commit() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}{'+dirty' if dirty else ''}" if sha else None


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _fmt_rate(stat: dict[str, Any] | None) -> str:
    if not stat or stat.get("rate") is None:
        return "n/a"
    lo, hi = stat["ci95"]
    share = f"{stat['successes']}/{stat['n']}"
    return f"{100 * stat['rate']:.1f}% [{100 * lo:.1f}-{100 * hi:.1f}] ({share})"


def _print_summary(report: dict[str, Any], paths: tuple[Path, Path]) -> None:
    s = report["summary"]
    print(f"run: {report['run']['name']}  questions: {s['n_questions']}  errors: {s['n_errors']}")
    ret = s.get("retrieval")
    if ret:
        print(f"  hit@{ret['k']}:       {_fmt_rate(ret['hit_at_k'])}")
        print(f"  complete@{ret['k']}:  {_fmt_rate(ret['complete_at_k'])}")
        print(f"  MRR@{ret['k']}:       {ret['mrr_at_k']['mean']:.3f}")
    ans = s.get("answer")
    if ans:
        print(f"  fully correct: {_fmt_rate(ans['fully_correct'])}")
        print(f"  answer correct (answerable): {_fmt_rate(ans['answerable']['answer_correct'])}")
        print(
            "  correct abstention (unanswerable): "
            f"{_fmt_rate(ans['unanswerable']['correct_abstention'])}"
        )
    nonzero = {d: c for d, c in s["diagnosis_counts"].items() if c}
    print(f"  diagnosis: {nonzero}")
    print(f"wrote {paths[0]} and {paths[1]}")


def _cmd_validate(args: argparse.Namespace) -> int:
    ds = load_dataset(args.dataset)
    cats: dict[str, int] = {}
    for q in ds.questions:
        cats[q.category] = cats.get(q.category, 0) + 1
    print(
        f"OK: {ds.name} v{ds.version}: {len(ds.questions)} questions, "
        f"{len(ds.corpus.documents)} documents, categories {cats}"
    )
    return EXIT_OK


def _cmd_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run = run_evaluation(
        config,
        resolve_dataset_dir(config, args.config),
        question_ids=args.questions,
        limit=args.limit,
    )
    report = build_report(run, git_commit=_git_commit(), config_file=args.config.as_posix())
    out_dir = args.out if args.out is not None else Path("reports") / config.name
    paths = write_report(report, out_dir)
    _print_summary(report, paths)
    return EXIT_QUESTION_ERRORS if report["summary"]["n_errors"] else EXIT_OK


def _load_report(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabError(f"cannot read report {path}: {exc}") from exc
    if not isinstance(data, dict) or "schema_version" not in data:
        raise LabError(f"{path} is not a rag-lab report")
    return data


def _cmd_report(args: argparse.Namespace) -> int:
    report = _load_report(args.report_json)
    out = args.out if args.out is not None else args.report_json.with_suffix(".md")
    out.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out}")
    return EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    report = _load_report(args.report_json)
    match = [q for q in report["questions"] if q["id"] == args.question_id]
    if not match:
        raise LabError(f"question {args.question_id!r} not in report")
    q = match[0]
    print(f"{q['id']} [{q['category']}] diagnosis={q['diagnosis']}")
    print(f"Q: {q['question']}")
    for ev in q["evidence"]:
        print(
            f"  evidence ({'retrieved' if ev['retrieved_at_k'] else 'MISSING'}): "
            f"{ev['doc']}: {ev['quote']}"
        )
    for p in q["retrieved"]:
        flag = "*" if p["in_context"] else " "
        print(
            f" {flag}#{p['rank']} {p['chunk_id']} score={p['score']} covers={p['covers_evidence']}"
        )
    if q["answer"]:
        print(f"A: {q['answer']['text']}")
        print(
            f"  correct={q['answer']['answer_correct']} "
            f"attribution={q['answer']['attribution_correct']}"
        )
    if q["error"]:
        print(f"ERROR ({q['error_stage']}): {q['error']}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-lab", description="RAG Reliability Lab")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-dataset", help="validate an evaluation dataset")
    p.add_argument("dataset", type=Path)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("eval", help="run an evaluation and write JSON + Markdown reports")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, help="output directory (default: reports/<run name>)")
    p.add_argument(
        "--question", dest="questions", action="append", help="only this question id (repeatable)"
    )
    p.add_argument("--limit", type=int, help="only the first N questions")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("report", help="re-render a Markdown report from report.json")
    p.add_argument("report_json", type=Path)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("show", help="print one question's retrieval and answer details")
    p.add_argument("report_json", type=Path)
    p.add_argument("question_id")
    p.set_defaults(func=_cmd_show)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except LabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
