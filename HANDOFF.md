# Handoff

_Last updated 2026-10-11. Work is on the local branch `rag-reliability-lab`, based on `main` @ `0d530d1`. **Nothing has been pushed**, and the original remote is untouched._

## Status

| Phase / milestone | State |
|---|---|
| Phase 1: audit | **Done**: [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md) (confirmed issues separated from assumptions; problems ranked) |
| Security fix G1 (`dockerignore` → `.dockerignore`) | **Done** (own commit) |
| Milestone 1: deterministic evaluation core | **Done** (this branch) |
| Milestone 2: configuration comparisons, LLM providers, replay | Next |
| Milestone 3: Streamlit inspector, Docker, rebuilt Q&A app, screenshots | Planned |
| Milestone 4: optional LLM judge (secondary signal), hosted-service write-up | Planned |

## Milestone 1: what changed

| Path | Change |
|---|---|
| `src/rag_reliability_lab/` | New typed package: ingestion with issue reporting, LangChain-parity chunker with exact offsets, BM25 and dense retrieval, extractive and Gemini generators, retrieval and answer metrics, Wilson and bootstrap CIs, failure diagnosis, JSON and Markdown reports, `rag-lab` CLI |
| `datasets/tarnwick-v1/` | Original fictional corpus (16 docs) + 56 questions across 7 categories + dataset card |
| `configs/` | `baseline-offline` (executed), `baseline-legacy-dense` and `legacy-dense-gemini` (provided, not executed) |
| `benchmarks/baseline-offline/` | Committed reference report, checked for reproducibility by the test suite |
| `tests/` | 136 tests: 135 run everywhere, 1 dense-integration test needs the `dense` extra |
| `docs/SCORING.md`, `docs/ARCHITECTURE.md` | Scoring definitions; architecture diagrams, including a future hosted-service sketch |
| `pyproject.toml`, `requirements/dev.lock` | Packaging; core dependency `pypdf` only; extras `dense`, `gemini`, `dev`; universal exact pins |
| `.github/workflows/ci.yml` | Lint, format, `mypy --strict`, dataset validation, tests, benchmark run + artifact, on PRs and pushes to `main` |
| `README.md`, `.gitignore` | Rewritten README; tooling outputs ignored |
| `app.py`, `streamlit_app.py`, `dockerfile`, `requirements.txt`, `data/` | **Unchanged** (legacy app; see audit) |

## Verification evidence (all executed 2026-10-11)

- `ruff check` / `ruff format --check`: clean. `mypy --strict`: no issues in 23 source files.
- `pytest -q`: **135 passed, 1 skipped** (Python 3.11.17). The skipped one is the dense integration test, which **passed** separately in a venv with sentence-transformers 6.1.0.
- **Windows line endings:** a copy with every dataset, config, benchmark and test file converted to CRLF (what Git for Windows checks out by default) passes all 135 tests. The `questions_sha256` hash is computed on newline-normalised text, and `.gitattributes` keeps LF everywhere.
- **The CI workflow was replayed step by step** in fresh venvs built only from `requirements/dev.lock`: it passes on Python **3.11.17 and 3.13.16**. `actionlint 1.7.12` reports no issues. Action majors (`@v7`) were checked against the actions' git tags.
- **Chunker parity** with LangChain `RecursiveCharacterTextSplitter` 1.1.3: **4,984 / 4,984** identical chunk texts (412 real PDF pages + 300 seeded random texts, 7 configurations). Offsets are identical to LangChain's `start_index` on real text (641 chunks). LangChain's search-based offsets ran backwards 21 times on random text; ours are exact by construction.
- **Reproducibility:** the baseline report is identical across repeated runs and across Python 3.11 and 3.13, excluding timing and environment fields.
- **Dense path:** ran end to end through `rag-lab eval` with the real sentence-transformers 6.1.0, using a tiny locally built random model (no download), and was deterministic across runs. The same `DenseRetriever` with a bag-of-words embedder reached hit@3 = 45/47, confirming the ranking code works.
- **Error handling:** tests inject retrieval and generation failures (each isolated per question, counted as failures, exit code 2), and Gemini 429 → 503 → success, 404 (no retry), exhausted retries, transport timeout, empty response and missing API key.

**Not verified in this environment** (blocked network, no key, nothing pushed):

- GitHub Actions running on GitHub
- any Docker build (Docker Hub blocked)
- retrieval with the real `all-MiniLM-L6-v2` (Hugging Face blocked)
- any real Gemini call (API blocked, no key); the provider is tested against a fake client only.

## Decisions needed from you

1. **Push / PR:** may I push `rag-reliability-lab` to GitHub and open a pull request? CI will then run there for the first time.
2. **License:** pick one for the code (e.g. MIT or Apache-2.0) and for the dataset (e.g. CC BY 4.0).
3. **Course PDFs** (`data/visualization`, Manipal University Jaipur course material, audit J1): remove them from the public repo? Purging them from history needs a force-push.
4. **Gemini spend:** running `configs/legacy-dense-gemini.toml` makes 56 paid calls per run. Confirm the model (`gemini-3.6-flash` per Google's deprecations page on 2026-10-11) and the budget.

## Next action

1. **(You, ~5 minutes, free)** On any machine with internet access:
   `pip install -r requirements/dev.lock && pip install -e ".[dense]" && rag-lab eval --config configs/baseline-legacy-dense.toml --out benchmarks/baseline-legacy-dense`.
   That gives the first real numbers for the legacy retriever (MiniLM), comparable to the committed BM25 baseline on the same dataset.
2. **(Me, Milestone 2)** Start with `rag-lab compare`: a sweep over chunk size × overlap × top_k × retriever, with **paired** statistics (exact McNemar on per-question outcomes, paired bootstrap for metric deltas). It should refuse to name a winner when a difference isn't significant, and use an embedding cache keyed by model and chunk hash so sweeps are cheap.

## Milestone 2 (proposed scope)

- Comparison runner, paired significance tests and a comparison report (JSON + Markdown).
- Replay cassettes: record LLM outputs keyed by prompt hash and re-score them offline. CI can then check LLM-run scoring without API calls.
- An OpenAI-compatible provider, which also covers local Ollama or vLLM for free LLM runs.
- Optional cross-encoder reranking (`dense` extra).
- Dataset v1.1: `distractor_facts`, values from a similar entity that must not be given as the answer (currently undetected; see `docs/SCORING.md` §6). Also a larger noise corpus so retrieval stops being near ceiling, and questions from a second author to reduce lexical overlap bias.

## Milestone 3 (planned)

- A Streamlit results inspector reading `report.json`: failures, retrieved passages vs missing evidence, and run-to-run diffs.
- A Dockerfile for the lab: CPU-only torch, non-root, `HEALTHCHECK`, and a CI job that builds it.
- Rebuild the legacy Q&A app on the lab's modules: index built at startup or build time, configurable model, citations shown, errors handled. This fixes audit D1, G2, B1, D2 and E1.
- Screenshots or a short demo script.

## Known issues carried over

- The legacy app does not work from a fresh clone (audit D1, G2, B1). Unchanged by design in M1.
- The deterministic scoring is alias-based. False negatives on paraphrases are possible, and extra wrong claims go undetected unless listed as stale facts.
- The dataset is small (n = 56) and single-author. Its intervals are wide; see the dataset card.
