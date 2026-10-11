# Benchmarks

Committed reports produced by `rag-lab eval`. Every value outside the `environment` block and the per-question `latency_ms` fields is deterministic. `tests/test_cli_and_report.py::test_golden_baseline_is_reproduced` re-runs `baseline-offline` in CI and fails if anything changes.

| Directory | Config | What it measures | Executed |
|---|---|---|---|
| `baseline-offline/` | `configs/baseline-offline.toml` | Legacy chunking (1000/200, k = 3) + BM25 + extractive answerer. A reproducible floor with no model and no API | Yes, 2026-10-11 (Linux, Python 3.11 and 3.13 give identical results) |
| *(pending)* `baseline-legacy-dense/` | `configs/baseline-legacy-dense.toml` | The legacy retriever exactly (MiniLM, exact L2, 1000/200, k = 3), retrieval only | **No.** Needs Hugging Face access |
| *(pending)* `legacy-dense-gemini/` | `configs/legacy-dense-gemini.toml` | Legacy retriever + Gemini with the grounded citation prompt | **No.** Needs an API key, and makes paid calls |

## Regenerating

```bash
rag-lab eval --config configs/baseline-offline.toml --out benchmarks/baseline-offline
pytest -q tests/test_cli_and_report.py   # confirms the committed report matches
```

Regenerate only after an intentional change to the code, dataset or config, and say why in the commit message. A changed `config_sha256`, `questions_sha256` or `corpus_fingerprint` makes the reason visible in the diff.

## Reading the numbers

- `n` is small (56 questions; 47 answerable, 9 unanswerable). Look at the 95% intervals before drawing any conclusion.
- Retrieval at 1000/200 is close to ceiling on this corpus (29 chunks). Answer quality is limited by the extractive baseline, which cannot combine facts or paraphrase.
- Latency values depend on the machine and are excluded from reproducibility checks.
