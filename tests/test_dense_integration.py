"""Integration test for the real sentence-transformers code path, with no download.

It builds a tiny, randomly initialised BERT + mean-pooling model locally. Its embeddings are
meaningless, so the test checks mechanics (encode kwargs, shapes, ranking, determinism),
not retrieval quality. Skipped unless the ``dense`` extra is installed.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from rag_reliability_lab.chunking import RecursiveCharacterSplitter, chunk_documents  # noqa: E402
from rag_reliability_lab.ingest import load_corpus  # noqa: E402
from rag_reliability_lab.retrieval.dense import (  # noqa: E402
    DenseRetriever,
    SentenceTransformerEmbedder,
)

from .conftest import DATASET_DIR  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from sentence_transformers import SentenceTransformer, models

    out = tmp_path_factory.mktemp("tiny-st")
    hf = out / "hf"
    hf.mkdir()
    text = " ".join(p.read_text(encoding="utf-8") for p in (DATASET_DIR / "corpus").rglob("*.md"))
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        *sorted(set(re.findall(r"[a-z0-9]+", text.lower()))),
    ]
    (hf / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=512,
    )  # fmt: skip
    transformers.BertModel(config).save_pretrained(hf)
    transformers.BertTokenizerFast(vocab_file=str(hf / "vocab.txt")).save_pretrained(hf)
    word = models.Transformer(str(hf), max_seq_length=256)
    pooling = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    SentenceTransformer(modules=[word, pooling], device="cpu").save(str(out / "st"))
    return out / "st"


def test_sentence_transformer_dense_retrieval_runs_and_is_deterministic(tiny_model: Path) -> None:
    chunks = chunk_documents(
        load_corpus(DATASET_DIR / "corpus").documents, RecursiveCharacterSplitter(1000, 200)
    )
    results = []
    for _ in range(2):
        retriever = DenseRetriever(SentenceTransformerEmbedder(str(tiny_model)), metric="l2")
        retriever.index(chunks)
        results.append(retriever.retrieve("What is the mileage rate?", 5))
    first, second = results
    assert [s.chunk.chunk_id for s in first] == [s.chunk.chunk_id for s in second]
    assert [s.rank for s in first] == [1, 2, 3, 4, 5]
    assert all(a.score >= b.score for a, b in pairwise(first))
    assert all(s.score <= 0 for s in first)  # negative squared L2 distance
