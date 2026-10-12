from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rag_reliability_lab import cli
from rag_reliability_lab.report import render_markdown

from .conftest import BASELINE_CONFIG, DATASET_DIR, REPO_ROOT, TINY_DOCS, TINY_QUESTIONS

GOLDEN_DIR = REPO_ROOT / "benchmarks" / "baseline-offline"


def strip_nondeterministic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: strip_nondeterministic(v)
            for k, v in value.items()
            if k not in ("environment", "latency_ms")
        }
    if isinstance(value, list):
        return [strip_nondeterministic(v) for v in value]
    return value


@pytest.fixture(scope="module")
def baseline_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("baseline")
    code = cli.main(["eval", "--config", str(BASELINE_CONFIG), "--out", str(out)])
    assert code == cli.EXIT_OK
    assert (out / "report.md").is_file()
    data: dict[str, Any] = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return data


def test_report_structure(baseline_report: dict[str, Any]) -> None:
    assert set(baseline_report) == {
        "schema_version", "run", "ingestion", "summary", "questions", "environment",
    }  # fmt: skip
    summary = baseline_report["summary"]
    assert summary["n_questions"] == 56 and summary["n_errors"] == 0
    assert sum(summary["diagnosis_counts"].values()) == 56
    assert summary["n_answerable"] + summary["n_unanswerable"] == 56
    q = baseline_report["questions"][0]
    assert {"evidence", "retrieved", "answer", "diagnosis", "retrieval"} <= set(q)
    assert all(p["text"] for p in q["retrieved"] if p["in_context"])


def test_markdown_has_all_sections(baseline_report: dict[str, Any]) -> None:
    md = render_markdown(baseline_report)
    for heading in (
        "## Retrieval quality",
        "## Answer quality",
        "## Diagnosis",
        "## By category",
        "## Failed questions",
        "## Ingestion",
        "## Reproduce",
    ):
        assert heading in md


def test_golden_baseline_is_reproduced(baseline_report: dict[str, Any]) -> None:
    """The committed benchmark must be byte-for-byte reproducible (apart from timings and the
    environment). If this fails after an intended change, regenerate with:
        rag-lab eval --config configs/baseline-offline.toml --out benchmarks/baseline-offline
    """
    golden = json.loads((GOLDEN_DIR / "report.json").read_text(encoding="utf-8"))
    assert strip_nondeterministic(baseline_report) == strip_nondeterministic(golden)


def test_validate_dataset_command(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert cli.main(["validate-dataset", str(DATASET_DIR)]) == cli.EXIT_OK
    assert "OK: tarnwick-v1" in capsys.readouterr().out
    assert cli.main(["validate-dataset", str(tmp_path)]) == cli.EXIT_INVALID
    assert "dataset.toml" in capsys.readouterr().err


def test_report_and_show_commands(
    baseline_report: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(baseline_report), encoding="utf-8")
    assert cli.main(["report", str(path)]) == cli.EXIT_OK
    assert (tmp_path / "report.md").read_text(encoding="utf-8") == render_markdown(baseline_report)
    assert cli.main(["show", str(path), "c01"]) == cli.EXIT_OK
    assert "c01 [conflicting]" in capsys.readouterr().out
    assert cli.main(["show", str(path), "zzz"]) == cli.EXIT_INVALID
    bogus = tmp_path / "bogus.json"
    bogus.write_text("[]")
    assert cli.main(["report", str(bogus)]) == cli.EXIT_INVALID


def test_eval_exit_code_two_when_questions_error(
    write_dataset: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rag_reliability_lab import runner

    class Broken:
        name = "broken"

        def generate(self, question: str, passages: object) -> object:
            raise RuntimeError("upstream 503")

        def describe(self) -> dict[str, object]:
            return {"provider": "broken"}

    monkeypatch.setattr(runner, "build_generator", lambda config: Broken())
    root = write_dataset(TINY_DOCS, TINY_QUESTIONS)
    config = tmp_path / "c.toml"
    config.write_text(f'name = "x"\ndataset = "{root.as_posix()}"\n', encoding="utf-8")
    out = tmp_path / "out"
    assert (
        cli.main(["eval", "--config", str(config), "--out", str(out)]) == cli.EXIT_QUESTION_ERRORS
    )
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["n_errors"] == 2
    assert "upstream 503" in (out / "report.md").read_text(encoding="utf-8")


def test_eval_rejects_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "c.toml"
    config.write_text('name = "x"\ndataset = "."\n[retrieval]\ntop_k = 0\n', encoding="utf-8")
    assert cli.main(["eval", "--config", str(config)]) == cli.EXIT_INVALID
    assert "top_k" in capsys.readouterr().err
