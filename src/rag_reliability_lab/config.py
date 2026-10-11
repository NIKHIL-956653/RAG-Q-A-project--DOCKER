"""Run configuration: strict TOML parsing into frozen, typed dataclasses.

Unknown keys, wrong types and out-of-range values raise ``ConfigError`` before any work
starts. Relative paths are resolved against the directory of the config file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigError


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: Literal["recursive_character"] = "recursive_character"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75


@dataclass(frozen=True)
class DenseConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    metric: Literal["l2", "cosine", "dot"] = "l2"
    normalize_embeddings: bool = False
    replace_newlines: bool = True
    device: str = "cpu"
    batch_size: int = 32


@dataclass(frozen=True)
class RetrievalConfig:
    method: Literal["bm25", "dense"] = "bm25"
    top_k: int = 3
    bm25: BM25Config = field(default_factory=BM25Config)
    dense: DenseConfig = field(default_factory=DenseConfig)


@dataclass(frozen=True)
class ExtractiveConfig:
    max_sentences: int = 2
    min_overlap: float = 0.5


@dataclass(frozen=True)
class GeminiConfig:
    model: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 512
    timeout_s: float = 60.0
    max_retries: int = 3
    backoff_s: float = 2.0
    api_key_env: str = "GOOGLE_API_KEY"


@dataclass(frozen=True)
class GenerationConfig:
    provider: Literal["extractive", "gemini", "none"] = "extractive"
    extractive: ExtractiveConfig = field(default_factory=ExtractiveConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


@dataclass(frozen=True)
class EvaluationConfig:
    evidence_overlap: float = 0.5
    cutoffs: tuple[int, ...] = (1, 3, 5, 10)
    bootstrap_resamples: int = 10_000
    seed: int = 0


@dataclass(frozen=True)
class RunConfig:
    name: str
    dataset: str
    description: str = ""
    min_chars: int = 1
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = _jsonable(dataclasses.asdict(self))
        return data

    def sha256(self) -> str:
        """Hash of the resolved configuration (canonical JSON). Two runs with the same hash
        used the same settings."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a table")
        return _build(hint, value, where)
    if origin is Literal:
        if value not in typing.get_args(hint):
            raise ConfigError(
                f"{where}: must be one of {list(typing.get_args(hint))}, got {value!r}"
            )
        return value
    if origin is tuple:
        (item_hint, _) = typing.get_args(hint)
        if not isinstance(value, list):
            raise ConfigError(f"{where}: expected a list")
        return tuple(_coerce(v, item_hint, f"{where}[{i}]") for i, v in enumerate(value))
    if origin in (typing.Union, types.UnionType):
        raise ConfigError(f"{where}: unsupported union type")  # pragma: no cover
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true or false")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string")
        return value
    raise ConfigError(f"{where}: unsupported type {hint!r}")  # pragma: no cover


def _build(cls: type[Any], data: Mapping[str, Any], where: str) -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {
        name: _coerce(value, hints[name], f"{where}.{name}" if where else name)
        for name, value in data.items()
    }
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where or 'config'}: {exc}") from exc


def validate(config: RunConfig) -> None:
    def check(ok: bool, message: str) -> None:
        if not ok:
            raise ConfigError(message)

    check(bool(config.name.strip()), "name must not be empty")
    check(config.min_chars >= 1, "min_chars must be >= 1")
    c = config.chunking
    check(c.chunk_size > 0, "chunking.chunk_size must be > 0")
    check(0 <= c.chunk_overlap < c.chunk_size, "chunking.chunk_overlap must be in [0, chunk_size)")
    r = config.retrieval
    check(1 <= r.top_k <= 100, "retrieval.top_k must be within [1, 100]")
    check(r.bm25.k1 >= 0 and 0 <= r.bm25.b <= 1, "retrieval.bm25: k1 >= 0 and 0 <= b <= 1")
    check(r.dense.batch_size >= 1, "retrieval.dense.batch_size must be >= 1")
    check(bool(r.dense.model.strip()), "retrieval.dense.model must not be empty")
    g = config.generation
    check(g.extractive.max_sentences >= 1, "generation.extractive.max_sentences must be >= 1")
    check(0 < g.extractive.min_overlap <= 1, "generation.extractive.min_overlap must be in (0, 1]")
    if g.provider == "gemini":
        check(
            bool(g.gemini.model.strip()),
            "generation.gemini.model must be set for provider 'gemini'",
        )
        check(
            g.gemini.timeout_s > 0 and g.gemini.max_retries >= 0,
            "generation.gemini: bad timeout/retries",
        )
        check(0.0 <= g.gemini.temperature <= 2.0, "generation.gemini.temperature must be in [0, 2]")
    e = config.evaluation
    check(0 < e.evidence_overlap <= 1, "evaluation.evidence_overlap must be in (0, 1]")
    check(
        bool(e.cutoffs) and all(1 <= k <= 100 for k in e.cutoffs),
        "evaluation.cutoffs must be in [1, 100]",
    )
    check(len(set(e.cutoffs)) == len(e.cutoffs), "evaluation.cutoffs must be unique")
    check(e.bootstrap_resamples >= 100, "evaluation.bootstrap_resamples must be >= 100")


def parse_config(data: Mapping[str, Any]) -> RunConfig:
    """Build and validate a config. The ``dataset`` path is kept exactly as written, so the
    config hash doesn't depend on the machine; resolve it with ``resolve_dataset_dir``."""
    if "name" not in data or "dataset" not in data:
        raise ConfigError("config must define 'name' and 'dataset'")
    config: RunConfig = _build(RunConfig, data, "")
    validate(config)
    return config


def load_config(path: Path) -> RunConfig:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    return parse_config(data)


def resolve_dataset_dir(config: RunConfig, config_path: Path | None) -> Path:
    """Relative dataset paths are relative to the config file (or the CWD if there is none)."""
    dataset = Path(config.dataset)
    if dataset.is_absolute():
        return dataset
    base = config_path.resolve().parent if config_path is not None else Path.cwd()
    return (base / dataset).resolve()
