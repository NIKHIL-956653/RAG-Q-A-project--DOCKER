from __future__ import annotations

import pytest

from rag_reliability_lab.errors import ConfigError
from rag_reliability_lab.generation import REFUSAL_TEXT, build_prompt
from rag_reliability_lab.generation.extractive import ExtractiveGenerator
from rag_reliability_lab.models import Chunk, ScoredChunk


def _passage(rank: int, text: str, doc: str = "doc.md") -> ScoredChunk:
    return ScoredChunk(
        Chunk(f"{doc}::c{rank:04d}", doc, rank, 0, len(text), text), 1.0 / rank, rank
    )


PASSAGES = [
    _passage(1, "The gateway buffers 72 hours of data. It has a red LED."),
    _passage(2, "The gateway buffers 72 hours of data. Unrelated sentence here."),
    _passage(3, "Sensors report every 15 minutes over LoRaWAN."),
]


def test_prompt_labels_passages_and_includes_refusal() -> None:
    prompt = build_prompt("  How long is data buffered? ", PASSAGES[:2])
    assert "[P1] source: doc.md\nThe gateway buffers" in prompt
    assert "[P2] source: doc.md" in prompt
    assert REFUSAL_TEXT in prompt
    assert prompt.rstrip().endswith("Question: How long is data buffered?\nAnswer:")
    assert "(no passages were retrieved)" in build_prompt("q", [])


def test_extractive_answers_with_best_sentence_and_citation() -> None:
    out = ExtractiveGenerator(max_sentences=2, min_overlap=0.5).generate(
        "How many hours of data does the gateway buffer?", PASSAGES
    )
    # the identical sentence in P2 is not repeated
    assert out.text.startswith("The gateway buffers 72 hours of data. [P1]")
    assert "[P2]" not in out.text


def test_extractive_abstains_below_threshold_and_without_passages() -> None:
    gen = ExtractiveGenerator(min_overlap=0.9)
    assert gen.generate("What colour is the gateway LED?", PASSAGES).text == REFUSAL_TEXT
    assert ExtractiveGenerator().generate("anything", []).text == REFUSAL_TEXT
    assert ExtractiveGenerator().generate("the of and", PASSAGES).text == REFUSAL_TEXT


def test_extractive_is_deterministic_and_validates() -> None:
    gen = ExtractiveGenerator()
    q = "How often do sensors report?"
    assert gen.generate(q, PASSAGES) == gen.generate(q, PASSAGES)
    with pytest.raises(ConfigError):
        ExtractiveGenerator(max_sentences=0)
    with pytest.raises(ConfigError):
        ExtractiveGenerator(min_overlap=0)
