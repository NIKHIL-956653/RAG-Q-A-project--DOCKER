from rag_reliability_lab.text import (
    contains_phrase,
    match_tokens,
    normalize_text,
    retrieval_tokens,
    split_sentences,
)


def test_normalize_unifies_quotes_dashes_and_case() -> None:
    raw = f"It{chr(0x2019)}s {chr(0x2212)}20 °C {chr(0x2013)} OK"  # curly quote, minus, en dash
    assert normalize_text(raw) == "it's -20 °c - ok"


def test_retrieval_tokens_drop_stopwords_and_split_codes() -> None:
    assert retrieval_tokens("What is the KX-220's IP rating?") == ["kx", "220", "ip", "rating"]


def test_match_tokens_keep_numbers_whole() -> None:
    assert match_tokens("EUR 0.34 per km, 1,500 total, 10:00") == [
        "eur", "0.34", "per", "km", "1,500", "total", "10:00",
    ]  # fmt: skip


def test_match_tokens_negative_numbers_but_not_ranges() -> None:
    assert match_tokens("-20 °C to 60 °C")[:3] == ["-20", "°", "c"]
    assert match_tokens("20-60") == ["20", "60"]
    assert match_tokens("KX220") == ["kx", "220"]
    assert match_tokens(chr(0x2212) + "40") == ["-40"]  # unicode minus sign


def test_contains_phrase_respects_token_boundaries() -> None:
    assert contains_phrase("within 14 days of start", "14 days")
    assert not contains_phrase("within 114 days of start", "14 days")
    assert not contains_phrase("within 14 calendar days", "14 days")
    assert contains_phrase("Price: €40.", "€40")
    assert not contains_phrase("anything", "")


def test_split_sentences_strips_markdown_and_splits() -> None:
    text = "# Title\n\n- First item. Second sentence!\n1. Numbered\n> quoted"
    assert split_sentences(text) == [
        "Title",
        "First item.",
        "Second sentence!",
        "Numbered",
        "quoted",
    ]
