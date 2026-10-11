"""Deterministic extractive baseline: answers with the retrieved sentences that best overlap
the question, or abstains.

It cannot paraphrase, combine facts or calculate. It is a fully reproducible lower bound and
a way to exercise the whole evaluation pipeline without any model or API.

Algorithm:

1. Split each retrieved passage into sentence units (``text.split_sentences``).
2. Score each unit by overlap: the share of the question's content tokens
   (``text.retrieval_tokens``) that appear in the unit.
3. If the best overlap is below ``min_overlap``, return the refusal sentence.
4. Otherwise return up to ``max_sentences`` distinct units with overlap >= ``min_overlap``,
   ordered by score, then passage rank, then position. Each unit is followed by its ``[P<n>]``
   label.

``min_overlap`` and ``max_sentences`` are fixed a priori (0.5 and 2). They were not tuned on
the evaluation dataset.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..errors import ConfigError
from ..models import ScoredChunk
from ..text import retrieval_tokens, split_sentences
from . import REFUSAL_TEXT, GeneratorOutput


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 2, min_overlap: float = 0.5) -> None:
        if max_sentences < 1:
            raise ConfigError(f"extractive.max_sentences must be >= 1, got {max_sentences}")
        if not 0.0 < min_overlap <= 1.0:
            raise ConfigError(f"extractive.min_overlap must be within (0, 1], got {min_overlap}")
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    def generate(self, question: str, passages: Sequence[ScoredChunk]) -> GeneratorOutput:
        q_terms = set(retrieval_tokens(question))
        candidates: list[tuple[float, int, int, str]] = []
        if q_terms:
            for p_label, passage in enumerate(passages, start=1):
                for s_index, sentence in enumerate(split_sentences(passage.chunk.text)):
                    overlap = len(q_terms & set(retrieval_tokens(sentence))) / len(q_terms)
                    candidates.append((overlap, p_label, s_index, sentence))
        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        chosen: list[tuple[int, str]] = []
        seen: set[str] = set()
        for overlap, p_label, _, sentence in candidates:
            if overlap < self.min_overlap or len(chosen) >= self.max_sentences:
                break
            if sentence in seen:
                continue
            seen.add(sentence)
            chosen.append((p_label, sentence))
        if not chosen:
            return GeneratorOutput(text=REFUSAL_TEXT, model=self.name, metadata={"abstained": True})
        text = " ".join(f"{sentence} [P{p_label}]" for p_label, sentence in chosen)
        best = candidates[0][0]
        return GeneratorOutput(
            text=text, model=self.name, metadata={"best_overlap": round(best, 6)}
        )

    def describe(self) -> dict[str, object]:
        return {
            "provider": "extractive",
            "max_sentences": self.max_sentences,
            "min_overlap": self.min_overlap,
        }
