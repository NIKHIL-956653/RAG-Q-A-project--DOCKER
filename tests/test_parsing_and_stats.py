from __future__ import annotations

import pytest

from rag_reliability_lab.generation import REFUSAL_TEXT
from rag_reliability_lab.metrics.parsing import detect_abstention, parse_citations
from rag_reliability_lab.metrics.stats import (
    bootstrap_mean_interval,
    mean,
    percentile,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("text", "labels", "malformed"),
    [
        ("Answer [P1].", (1,), ()),
        ("A [P2][P1] and again [P2].", (2, 1), ()),
        ("A [P1, P3] b [P2; P4]", (1, 3, 2, 4), ()),
        ("Lowercase [p5] and spaced [ P 6 ]", (5, 6), ()),
        ("Ordinary [sic] brackets [1] are ignored", (), ()),
        ("Mixed [P1, see below]", (1,), ("see below",)),
        ("No citations at all", (), ()),
    ],
)
def test_parse_citations(text: str, labels: tuple[int, ...], malformed: tuple[str, ...]) -> None:
    parsed = parse_citations(text)
    assert parsed.labels == labels
    assert parsed.malformed == malformed


@pytest.mark.parametrize(
    ("text", "pattern"),
    [
        (REFUSAL_TEXT, "refusal_text"),
        ("I couldn't find that information.", "could_not_find"),
        ("The price is not mentioned in the provided documents.", "not_in_sources"),
        ("The passages do not contain the CEO's name.", "sources_do_not"),
        ("There is no information about parental leave.", "no_information"),
        ("The KX-200 has a 2-year warranty [P1].", None),
        ("Not all sensors support LTE-M [P2].", None),
    ],
)
def test_detect_abstention(text: str, pattern: str | None) -> None:
    assert detect_abstention(text) == pattern


def test_wilson_interval_reference_values() -> None:
    lo, hi = wilson_interval(5, 10)  # type: ignore[misc]
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    lo, hi = wilson_interval(0, 10)  # type: ignore[misc]
    assert lo == 0.0 and round(hi, 4) == 0.2775
    lo, hi = wilson_interval(10, 10)  # type: ignore[misc]
    assert round(lo, 4) == 0.7225 and hi == 1.0
    assert wilson_interval(0, 0) is None
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_bootstrap_is_seeded_and_brackets_the_mean() -> None:
    values = [0.0, 0.5, 1.0, 1.0, 0.25, 0.75, 1.0, 0.0]
    a = bootstrap_mean_interval(values, resamples=2000, seed=1)
    b = bootstrap_mean_interval(values, resamples=2000, seed=1)
    assert a == b
    assert a is not None and a[0] <= (mean(values) or 0) <= a[1]
    assert bootstrap_mean_interval([], resamples=10) is None
    assert bootstrap_mean_interval([0.5] * 5, resamples=100) == (0.5, 0.5)


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) is None
