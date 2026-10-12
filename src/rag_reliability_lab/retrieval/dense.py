"""Dense retrieval with exact (brute-force) vector search.

Defaults mirror the legacy pipeline (audit section C): ``HuggingFaceEmbeddings`` with
``sentence-transformers/all-MiniLM-L6-v2``, newlines replaced by spaces before encoding,
no embedding normalisation, and a FAISS ``IndexFlatL2`` (exact squared-L2 search). Exact
search over the same vectors gives the same ranking as FAISS flat search up to
floating-point ties, so FAISS isn't needed at evaluation scale.

``numpy`` and ``sentence-transformers`` are optional (``pip install .[dense]``) and are
imported lazily, so the core package works without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from ..errors import ConfigError
from ..models import Chunk, ScoredChunk
from . import tie_break_key

Metric = Literal["l2", "cosine", "dot"]


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> Any: ...  # returns a 2-D float array


class SentenceTransformerEmbedder:
    """Wraps ``sentence_transformers.SentenceTransformer`` (requires the ``dense`` extra)."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        replace_newlines: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ConfigError(
                "dense retrieval needs the optional extra: pip install 'rag-reliability-lab[dense]'"
            ) from exc
        self.name = model
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.replace_newlines = replace_newlines
        self._model = SentenceTransformer(model, device=device)

    def embed(self, texts: Sequence[str]) -> Any:
        prepared = [t.replace("\n", " ") for t in texts] if self.replace_newlines else list(texts)
        return self._model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


class DenseRetriever:
    name = "dense"

    def __init__(self, embedder: Embedder, metric: Metric = "l2") -> None:
        if metric not in ("l2", "cosine", "dot"):
            raise ConfigError(f"unknown dense metric {metric!r}")
        self.embedder = embedder
        self.metric: Metric = metric
        self.score_kind = {"l2": "neg_squared_l2", "cosine": "cosine", "dot": "dot"}[metric]
        self._chunks: list[Chunk] = []
        self._matrix: Any = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        import numpy as np

        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return
        matrix = np.asarray(self.embedder.embed([c.text for c in self._chunks]), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self._chunks):
            raise ConfigError(f"embedder returned shape {matrix.shape}, expected (n_chunks, dim)")
        if self.metric == "cosine":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = (matrix / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)
        self._matrix = matrix

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        import numpy as np

        if k < 1:
            raise ConfigError(f"k must be >= 1, got {k}")
        if not self._chunks:
            return []
        q = np.asarray(self.embedder.embed([query]), dtype=np.float32).reshape(-1)
        if self.metric == "l2":
            diff = self._matrix - q
            scores = -np.einsum("ij,ij->i", diff, diff)
        else:
            if self.metric == "cosine":
                q_norm = float(np.linalg.norm(q))
                q = q / (q_norm if q_norm else 1.0)
            scores = self._matrix @ q
        order = sorted(
            range(len(self._chunks)),
            key=lambda i: (-float(scores[i]), tie_break_key(self._chunks[i])),
        )[:k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), rank=r)
            for r, i in enumerate(order, start=1)
        ]

    def describe(self) -> dict[str, object]:
        return {"method": "dense", "model": self.embedder.name, "metric": self.metric}
