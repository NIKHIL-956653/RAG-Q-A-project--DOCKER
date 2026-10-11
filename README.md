# RAG Reliability Lab

Measure, debug and improve retrieval-augmented generation (RAG) with **reproducible, deterministic evaluations**. It scores retrieval and answers separately, gives every rate a confidence interval, and names the pipeline stage that failed for every question.

> **Status: Milestone 1 (evaluation core).** This repository started as a Docker-packaged RAG Q&A demo. Its audit is in [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md); progress and next steps are in [`HANDOFF.md`](HANDOFF.md).

## What it does

- **A version-controlled evaluation dataset** ([`datasets/tarnwick-v1`](datasets/tarnwick-v1/README.md)): 16 documents and 56 questions covering factual, multi-passage, unanswerable, distractor, conflicting, poorly phrased and attribution cases. Evidence is annotated as exact quotes, so the labels work for any chunk size. A strict validator proves quotes exist, facts are grounded, and "unanswerable" questions really are unanswerable.
- **Retrieval metrics**: hit@k, complete@k, recall@k, precision@k and MRR@k against span-level evidence, at the system's `top_k` plus deeper diagnostic cutoffs.
- **Answer metrics, deterministic**: required-fact coverage, abstention on unanswerable questions, stale-value detection for conflicting sources, citation validity, precision and evidence recall, and a fact-level grounding check. There is no LLM judge. Definitions are in [`docs/SCORING.md`](docs/SCORING.md).
- **Failure diagnosis**: each question is labelled `pass`, `retrieval_miss`, `false_abstention`, `stale_answer`, `generation_miss`, `citation_error`, `hallucinated_answer` or `error`.
- **Reports**: a JSON report (the source of truth) and a Markdown report with 95% CIs, failed questions with their retrieved passages and missing evidence, latency, errors and reproducibility hashes.
- **Offline by default**: a BM25 retriever and an extractive answerer run anywhere, with no model download, no API key and no cost. Dense retrieval (sentence-transformers) and Gemini are optional extras.

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements/dev.lock   # exact pins, tested on Python 3.11 and 3.13
pip install --no-deps -e .

rag-lab validate-dataset datasets/tarnwick-v1
rag-lab eval --config configs/baseline-offline.toml        # writes reports/baseline-offline/
rag-lab show reports/baseline-offline/report.json c01      # inspect one question
pytest -q                                                  # 134 tests (+1 with the dense extra)
```

The `eval` exit code is 0 on success, 1 for invalid input (config, dataset or arguments), and 2 if any question raised an error. The report is written in every case.

## Baseline results

[`benchmarks/baseline-offline/`](benchmarks/baseline-offline/report.md) holds the offline baseline: the legacy app's chunking (1000/200, k = 3) with **BM25 standing in for MiniLM and an extractive answerer standing in for Gemini**. It is a reproducible floor, **not** a measurement of the legacy app. CI re-runs it and fails if any deterministic value changes.

| Metric (n) | Value | 95% CI |
|---|---|---|
| Hit@3 (47 answerable) | 97.9% | 88.9–99.6% |
| MRR@3 (47) | 0.876 | 0.801–0.943 |
| Fully correct (56) | 57.1% | 44.1–69.2% |
| Answer correct (47 answerable) | 55.3% | 41.2–68.6% |
| Correct abstention (9 unanswerable) | 66.7% | 35.4–87.9% |

Retrieval is near ceiling on this small corpus, and the extractive answerer is the bottleneck: 13 of its 24 failures are abstentions with the evidence already retrieved. The intervals are wide on purpose, because 56 questions cannot separate configurations that differ by a few points. See the [dataset card](datasets/tarnwick-v1/README.md#known-limitations-and-biases).

## Evaluating other configurations

| Config | Retriever | Generator | Needs |
|---|---|---|---|
| `configs/baseline-offline.toml` | BM25 | extractive | nothing |
| `configs/baseline-legacy-dense.toml` | legacy MiniLM + exact L2 (FAISS-equivalent) | none (retrieval only) | `pip install -e ".[dense]"` and Hugging Face access |
| `configs/legacy-dense-gemini.toml` | legacy MiniLM | Gemini, grounded citation prompt | `.[dense,gemini]`, `GOOGLE_API_KEY`, **paid calls** (try `--limit 5`) |

The last two have **not yet been run** to produce committed results: the environment that built this milestone had no access to Hugging Face or the Gemini API. The original model `gemini-2.0-flash` was shut down by Google on 2026-06-01; the config uses the replacement Google listed on 2026-10-11.

To add your own dataset, copy the layout of `datasets/tarnwick-v1` (`dataset.toml`, `corpus/`, `questions.jsonl`), run `rag-lab validate-dataset`, and point a config's `dataset` at it. PDFs are loaded one document per page.

## Repository layout

```
src/rag_reliability_lab/   the package (see docs/ARCHITECTURE.md)
datasets/tarnwick-v1/      evaluation dataset and dataset card
configs/                   run configurations (TOML)
benchmarks/                committed, reproducible reports
docs/                      SCORING.md, ARCHITECTURE.md
tests/                     unit, LangChain parity, golden benchmark, dense integration
requirements/dev.lock      exact dependency pins (universal, Python >= 3.11)
.github/workflows/ci.yml   lint, mypy --strict, tests and benchmark on every PR
app.py, streamlit_app.py,  legacy Q&A app (see below)
dockerfile, requirements.txt
```

## Legacy Q&A app

The original app (`app.py` builds a FAISS index from PDFs; `streamlit_app.py` answers questions with Gemini) is unchanged in this milestone, apart from fixing the ignored `.dockerignore`. As audited on 2026-10-11 it **does not work from a fresh clone**:

- its Gemini model has been shut down;
- `app.py` reads a hardcoded Windows path;
- the Docker image needs a `vector_db/` that is never built;
- two of the 14 course PDFs never reach the index (one is corrupt, one is in a sub-folder).

Details and evidence are in [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md). Milestone 3 rebuilds it on the lab's modules with Docker support.

## Limitations

- The dataset is small and single-author (see its card). It is built for finding failure modes and catching regressions, not for ranking models.
- Fact matching is alias-based. Correct paraphrases without an alias are scored as wrong, and extra wrong claims are only caught when listed as stale facts.
- Comparing configurations (sweeps with paired significance tests), replayable LLM runs, a Streamlit inspector and a Docker image for the lab are planned, not built ([`HANDOFF.md`](HANDOFF.md)).

## License

Not yet chosen (see `HANDOFF.md`). The course PDFs under `data/` are third-party material; see audit finding J1.

Built by Nikhil Chandra.
