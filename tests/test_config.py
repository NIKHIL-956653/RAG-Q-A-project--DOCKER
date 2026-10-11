from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_reliability_lab.config import (
    load_config,
    parse_config,
    resolve_dataset_dir,
)
from rag_reliability_lab.errors import ConfigError

from .conftest import BASELINE_CONFIG, DATASET_DIR, REPO_ROOT

BASE: dict[str, Any] = {"name": "t", "dataset": "data"}


def test_defaults_match_legacy_baseline() -> None:
    cfg = parse_config(BASE)
    assert (cfg.chunking.chunk_size, cfg.chunking.chunk_overlap, cfg.retrieval.top_k) == (
        1000,
        200,
        3,
    )
    assert cfg.retrieval.dense.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retrieval.dense.metric == "l2" and cfg.retrieval.dense.replace_newlines


@pytest.mark.parametrize(
    "override",
    [
        {"chunking": {"chunk_size": 100, "chunk_overlap": 100}},
        {"chunking": {"chunk_size": "1000"}},
        {"chunking": {"chunk_sizes": 10}},
        {"retrieval": {"top_k": 0}},
        {"retrieval": {"method": "faiss"}},
        {"retrieval": {"bm25": {"b": 2}}},
        {"generation": {"provider": "gemini"}},  # model not set
        {"generation": {"extractive": {"min_overlap": 0}}},
        {"evaluation": {"cutoffs": [3, 3]}},
        {"evaluation": {"evidence_overlap": 1.5}},
        {"evaluation": {"bootstrap_resamples": 10}},
        {"name": " "},
        {"unexpected": True},
        {"retrieval": {"top_k": True}},
    ],
)
def test_invalid_configs_rejected(override: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        parse_config({**BASE, **override})


def test_name_and_dataset_required() -> None:
    with pytest.raises(ConfigError, match="name"):
        parse_config({"dataset": "x"})


def test_int_accepted_for_float_fields() -> None:
    cfg = parse_config({**BASE, "retrieval": {"bm25": {"k1": 2}}})
    assert cfg.retrieval.bm25.k1 == 2.0 and isinstance(cfg.retrieval.bm25.k1, float)


def test_hash_is_stable_and_sensitive() -> None:
    a = parse_config(BASE).sha256()
    assert a == parse_config(dict(BASE)).sha256()
    assert a != parse_config({**BASE, "retrieval": {"top_k": 4}}).sha256()


def test_shipped_config_resolves_dataset_relative_to_file() -> None:
    cfg = load_config(BASELINE_CONFIG)
    assert cfg.dataset == "../datasets/tarnwick-v1"  # kept as written: hash is machine-independent
    assert resolve_dataset_dir(cfg, BASELINE_CONFIG) == DATASET_DIR


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "configs").glob("*.toml")))
def test_all_shipped_configs_parse(path: Path) -> None:
    load_config(path)


def test_bad_toml_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("name = [unterminated")
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")
