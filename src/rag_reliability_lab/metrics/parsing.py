"""Parse citations and detect abstentions in generated answers (rule-based, auditable)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..text import normalize_text

_BRACKET_RE = re.compile(r"\[([^\[\]]{1,60})\]")
_LABEL_RE = re.compile(r"^\s*P\s*(\d{1,4})\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Citations:
    labels: tuple[int, ...]  # unique passage labels in order of first appearance
    malformed: tuple[str, ...]  # bracket parts that look like citations but don't parse


def parse_citations(text: str) -> Citations:
    """Extract ``[P1]``, ``[P1][P3]``, ``[P1, P3]`` and ``[P2; P4]`` markers.

    A bracket group counts as a citation group only if at least one part parses as ``P<n>``.
    Other parts of that group are reported as malformed. Ordinary bracketed text such as
    ``[sic]`` is ignored.
    """
    labels: list[int] = []
    malformed: list[str] = []
    for group in _BRACKET_RE.findall(text):
        parts = [p for p in re.split(r"[,;]", group) if p.strip()]
        matches = [_LABEL_RE.match(p) for p in parts]
        if not any(matches):
            continue
        for part, match in zip(parts, matches, strict=True):
            if match:
                label = int(match.group(1))
                if label not in labels:
                    labels.append(label)
            else:
                malformed.append(part.strip())
    return Citations(labels=tuple(labels), malformed=tuple(malformed))


# Patterns are checked in order against the normalised (casefolded) answer. The first one is
# the exact refusal sentence the prompt asks for; the rest cover common LLM paraphrases.
ABSTENTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("refusal_text", re.compile(r"could not find the answer in the provided documents")),
    ("could_not_find", re.compile(r"\b(?:could ?not|couldn't|can ?not|can't|cannot) find\b")),
    (
        "not_in_sources",
        re.compile(
            r"\bnot (?:mentioned|stated|specified|provided|included|covered|found|available)\b"
            r".{0,40}\b(?:documents?|passages?|context|sources?)\b"
        ),
    ),
    (
        "sources_do_not",
        re.compile(
            r"\b(?:documents?|passages?|context|sources?)\b.{0,30}\b(?:do|does|did) ?(?:not|n't) "
            r"(?:contain|mention|state|specify|provide|include|say|cover)\b"
        ),
    ),
    ("no_information", re.compile(r"\bno (?:information|mention|details?)\b")),
)


def detect_abstention(text: str) -> str | None:
    """Return the name of the first matching abstention pattern, or ``None``."""
    normalized = " ".join(normalize_text(text).split())
    for name, pattern in ABSTENTION_PATTERNS:
        if pattern.search(normalized):
            return name
    return None
