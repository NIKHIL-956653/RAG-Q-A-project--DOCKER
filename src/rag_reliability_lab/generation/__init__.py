"""Answer generators. Every generator returns plain text with ``[P<n>]`` citation markers.

Citations and abstentions are parsed from that text by the scorer, the same way for every
generator, so an LLM gets no special treatment over the deterministic baseline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..models import ScoredChunk

PROMPT_VERSION = "grounded-qa-v1"
REFUSAL_TEXT = "I could not find the answer in the provided documents."

PROMPT_TEMPLATE = """You answer questions using only the numbered passages below.

Rules:
1. Use only information stated in the passages. Do not use outside knowledge.
2. Cite the passage label for every claim, for example [P1] or [P2][P3].
3. If the passages do not contain the answer, reply exactly: {refusal}
4. If passages disagree, prefer the source that is explicitly newer or superseding, and mention the disagreement.

Passages:
{context}

Question: {question}
Answer:"""

PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def build_prompt(question: str, passages: Sequence[ScoredChunk]) -> str:
    """Render passages with stable labels P1..Pn and their source document ids."""
    blocks = [
        f"[P{i}] source: {sc.chunk.doc_id}\n{sc.chunk.text}"
        for i, sc in enumerate(passages, start=1)
    ]
    context = "\n\n".join(blocks) if blocks else "(no passages were retrieved)"
    return PROMPT_TEMPLATE.format(refusal=REFUSAL_TEXT, context=context, question=question.strip())


@dataclass(frozen=True, slots=True)
class GeneratorOutput:
    text: str
    model: str
    metadata: Mapping[str, object] = field(default_factory=dict)


class Generator(Protocol):
    name: str

    def generate(self, question: str, passages: Sequence[ScoredChunk]) -> GeneratorOutput: ...

    def describe(self) -> dict[str, object]: ...
