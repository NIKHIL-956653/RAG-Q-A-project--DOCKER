"""Deterministic text normalisation and tokenisation.

Two tokenisers serve different purposes:

* ``retrieval_tokens`` -- lowercase words and digit runs with English stopwords removed.
  Used by BM25 and by the extractive baseline's overlap score.
* ``match_tokens`` -- keeps numbers whole (``0.34``, ``1,500``, ``-20``), keeps stopwords,
  and treats currency, percent and degree symbols as tokens. Used for fact matching, where
  "14 days" must not match "114 days".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_QUOTE_MAP = str.maketrans(
    {
        chr(0x2018): "'",  # left single quotation mark
        chr(0x2019): "'",  # right single quotation mark
        chr(0x201C): '"',  # left double quotation mark
        chr(0x201D): '"',  # right double quotation mark
        chr(0x2212): "-",  # minus sign
        chr(0x2013): "-",  # en dash
        chr(0x2014): "-",  # em dash
        chr(0x00A0): " ",  # no-break space
    }
)

_RETRIEVAL_TOKEN_RE = re.compile(r"[^\W_]+")
_MATCH_TOKEN_RE = re.compile(r"(?:(?<!\w)-)?\d+(?:[.,:]\d+)*|[^\W\d_]+|[€$£%°]")

# A compact English stopword list (articles, pronouns, auxiliaries, prepositions, conjunctions).
STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its
    itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves
    s t don shall may might must also per via
    """.split()  # noqa: SIM905 - readable word list
)


def normalize_text(text: str) -> str:
    """NFKC-normalise, unify quotes and dashes, and casefold."""
    return unicodedata.normalize("NFKC", text).translate(_QUOTE_MAP).casefold()


def retrieval_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed (order preserved)."""
    return [t for t in _RETRIEVAL_TOKEN_RE.findall(normalize_text(text)) if t not in STOPWORDS]


def match_tokens(text: str) -> list[str]:
    """Tokens for phrase matching: whole numbers, words and unit symbols; nothing removed."""
    return _MATCH_TOKEN_RE.findall(normalize_text(text))


def contains_token_sequence(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    """True if ``needle`` occurs as a contiguous run inside ``haystack``."""
    n = len(needle)
    if n == 0:
        return False
    first = needle[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and list(haystack[i : i + n]) == list(needle):
            return True
    return False


def contains_phrase(text: str, phrase: str) -> bool:
    """Token-level phrase containment, insensitive to case, punctuation and spacing."""
    return contains_token_sequence(match_tokens(text), match_tokens(phrase))


_MD_PREFIX_RE = re.compile(r"^(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*)")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-like units. Each line is split on terminal punctuation,
    and Markdown heading, bullet and quote markers are removed."""
    units: list[str] = []
    for raw_line in text.split("\n"):
        line = _MD_PREFIX_RE.sub("", raw_line.strip()).strip()
        if not line:
            continue
        units.extend(part.strip() for part in _SENTENCE_BOUNDARY_RE.split(line) if part.strip())
    return units
