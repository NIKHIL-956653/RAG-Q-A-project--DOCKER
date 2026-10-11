# Project Audit — RAG-Q-A-project--DOCKER

| | |
|---|---|
| Audited commit | `0d530d1` (`main`, 2026-03-24), 4 commits total |
| Audit date | 2026-10-11 |
| Audit environment | Linux x86_64 cloud workspace, Python 3.10.20 venv (matches the Dockerfile's `python:3.10-slim`), pip 26.2.1, Docker 29.8.2 (daemon started locally) |
| Network limits | PyPI and GitHub reachable. **Hugging Face Hub, Docker Hub, download.pytorch.org and the Gemini API were blocked** in this environment. No Gemini API key was used. |

## Evidence legend

- **[EXECUTED]** — reproduced by running code or commands in this audit. Commands are listed in the appendix.
- **[INSPECTED]** — read from source; not executed.
- **[VENDOR DOC]** — taken from vendor documentation fetched on 2026-10-11.
- **[ASSUMPTION]** — inferred and not verified. All of these are collected in their own section.

## What was and was not executed

| Item | Status |
|---|---|
| `pip install -r requirements.txt` (Python 3.10, unpinned, resolved 2026-10-11) | **[EXECUTED]** succeeded in 322 s; 153 packages; venv 6.4 GB |
| `python app.py` as committed | **[EXECUTED]** fails immediately (hardcoded Windows path) |
| `app.py` with only the folder path changed | **[EXECUTED]** PDF loading + chunking ran; failed at embedding-model download (Hugging Face blocked here) |
| What the LLM receives (same prompt/chain wiring, fake retriever + fake LLM) | **[EXECUTED]** |
| `streamlit_app.py` on a fresh clone (missing `vector_db/`) | **[EXECUTED]** the failing call (`FAISS.load_local`) in isolation |
| Docker build-context contents (`FROM scratch` probe, no base-image pull) | **[EXECUTED]** |
| Building/running the real Docker image | **Not executed**: Docker Hub blocked (`python:3.10-slim` cannot be pulled) |
| Gemini generation | **Not executed**: API blocked, no key |
| Dense retrieval with `all-MiniLM-L6-v2` | **Not executed**: Hugging Face blocked |
| Existing tests / CI | None exist (no `tests/`, no `.github/`) |

## Architecture as found

```
app.py (one-off script, run manually)            streamlit_app.py (Docker CMD)
  PyPDFLoader(<hardcoded C:\ path>)                 HuggingFaceEmbeddings(all-MiniLM-L6-v2)
  -> RecursiveCharacterTextSplitter(1000, 200)      FAISS.load_local("vector_db")  <- must already exist
  -> HuggingFaceEmbeddings(all-MiniLM-L6-v2)        retriever k=3 -> prompt -> Gemini 2.0 Flash (t=0.3)
  -> FAISS.from_documents -> save "vector_db/"      -> answer text only (no sources)
  -> one hardcoded test question to Gemini
```

There is no shared module: model names, `k`, the prompt and the LLM settings are copy-pasted between the two files. All parameters are hardcoded except `GOOGLE_API_KEY`.

## Confirmed issues

Severity: **Critical** = the app is broken or a secret leaks. **High** = silent wrong results or no way to verify quality. **Medium** = reliability, maintainability or cost. **Low** = hygiene.

### Security and Docker

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| G1 | **Critical** | `dockerignore` is missing its leading dot, so Docker ignores it. `COPY . .` copies `.env` (the API key), `vector_db/`, `venv/` and everything else into the image. Any image built from a working copy that contains `.env` holds the key in a layer. | **[EXECUTED]** A `FROM scratch` + `COPY . /ctx` probe with a dummy `.env` produced an image containing `ctx/.env`, `ctx/vector_db/`, `ctx/venv/`. After renaming the file to `.dockerignore`, all three were excluded. |
| G2 | **Critical** | A fresh clone cannot produce a working container. `vector_db/` is gitignored, the image never builds it, and the only builder (`app.py`) can't run (B1). The image can only work if a host-built `vector_db/` leaks in through G1, so fixing G1 alone breaks that path. | **[EXECUTED]** `FAISS.load_local("vector_db", …)` without the folder raises `RuntimeError … could not open vector_db/index.faiss for reading`. |
| G3 | Medium | Container runs as root, has no `HEALTHCHECK` and no model cache (the embedding model is re-downloaded in every fresh container). The base image tag is unpinned and Python 3.10 reaches end-of-life in October 2026 (PEP 619). | **[INSPECTED]** |
| G4 | Info | The lowercase `dockerfile` works: `docker build .` falls back to it. Some linters and IDEs won't detect it. | **[EXECUTED]** |

### Model integration and prompting

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| D1 | **Critical** | The hardcoded `gemini-2.0-flash` was **shut down on 2026-06-01**. Google lists `gemini-3.6-flash` as the replacement. Every question should fail today. | **[VENDOR DOC]** ai.google.dev/gemini-api/docs/deprecations. No API call was made. |
| D2 | High | The context is the **Python repr of a `List[Document]`**; there is no formatting step. The LLM receives `[Document(metadata={'producer': …, 'author': 'ashwi', 'source': '/abs/path/…', …}, page_content="…\n…")]`. That wastes tokens, sends local paths and PDF author metadata to a third-party API, and gives the model no passage labels it could cite. | **[EXECUTED]** The captured prompt contains `Document(`, `metadata=` and literal `\n` escapes. |
| D3 | Medium | No timeout, retry or max-token settings. Temperature 0.3 is hardcoded, which adds variance to any evaluation. | **[INSPECTED]** |
| E1 | High | **No source citations.** The chain discards the retrieved documents, the prompt never asks for citations, and the UI shows only the answer text, so there is no way to check which passage supported an answer. | **[INSPECTED]** |

### Ingestion and chunking

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| B1 | **Critical** | `app.py` hardcodes `C:\Users\nirun\Desktop\rag project!\data\visualization`. The index cannot be built anywhere else. | **[EXECUTED]** `FileNotFoundError` on line 20. |
| B2 | High | Non-recursive `os.listdir` silently skips `unit2/DADS304 Unit-02.pdf` (29 pages). | **[EXECUTED]** 418 of 447 pages loaded; Unit-02 contributes 0 chunks. |
| B3 | High | `DADS304 Unit-01.pdf` is **corrupt in the repository**: no text from any of its 35 pages. Unit 1 is presumably the introductory unit, which `app.py`'s own test question ("What is data visualization?") points to. | **[EXECUTED]** pypdf extracts 0 characters ("incorrect header check"); poppler `pdftotext` also fails ("Bad FCHECK in flate stream"). |
| B4 | High | No ingestion report. Empty or failed documents silently yield 0 chunks. With B2 and B3, **64 of 447 pages (14%) are missing from the index** without any warning. | **[EXECUTED]** Per-file chunk counts: Unit-01 = 0, Unit-02 = 0, the other 12 = 30–69 each. 594 chunks total. |
| B5 | Low | Files are processed in filesystem order (unsorted), so chunk order and FAISS ids differ across machines. | **[EXECUTED]** `os.listdir` returned `Unit-07, Unit-06, Unit-10, …` |
| B6 | Low | Chunks never cross page boundaries (per-page splitting), and every page repeats the header `DADS304: Visualization … Manipal University Jaipur (MUJ)`. That boilerplate lands in every chunk embedding. | **[EXECUTED]** observed in extracted text |

### Retrieval

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| C1 | High | **The index can go stale.** If `vector_db/` exists it is loaded regardless of changes to the corpus, chunking or embedding model. There is no manifest or hash. `app.py` still re-loads and re-splits every PDF even when it then loads the old index. Any experiment comparing configurations would silently reuse old vectors. | **[INSPECTED]** |
| C2 | Medium | Fixed `k=3`, no score threshold, no reranking. The retriever always returns 3 chunks, even for questions the corpus cannot answer. | **[INSPECTED]** |
| C3 | Medium | The embedding model is downloaded from Hugging Face at runtime, and there is no offline mode. | **[EXECUTED]** The download failed here with proxy 403, which is an environment limit. |
| C4 | Low | `allow_dangerous_deserialization=True` (pickle). Acceptable only for self-built indexes; G1 lets a host index enter the image unchecked. | **[INSPECTED]** |

### Dependencies

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| A1 | High | **Nothing is pinned.** On 2026-10-11 the requirements resolved to langchain 1.4.4, langchain-core 1.6.9, langchain-community 0.4.2, langchain-google-genai 4.4.1, sentence-transformers 6.1.0, transformers 5.19.0, torch 2.14.1 and faiss-cpu 1.15.1. A build on another day gets a different stack. | **[EXECUTED]** full freeze saved during the audit |
| A2 | Medium | `requirements.txt` is **UTF-16 LE with BOM and CRLF**. pip 26.2.1 and uv both parse it, but git treats it as binary (`git log --stat`: `Bin 0 -> 370 bytes`), so dependency changes can't be reviewed in diffs. | **[EXECUTED]** |
| A3 | Medium | A CPU-only app installs the CUDA stack: 16 `nvidia-*`/`triton` packages. torch 1.2 GB + nvidia 3.2 GB + triton 0.9 GB = **5.3 GB of the 6.4 GB venv**. | **[EXECUTED]** |
| A4 | Medium | `langchain_community.embeddings.HuggingFaceEmbeddings` emits `LangChainDeprecationWarning` ("deprecated in 0.2.2 and will be removed in 1.0"). Without pins, a future release can break the import. | **[EXECUTED]** |
| A5 | Low | `google-generativeai` (deprecated SDK) is installed but unused. langchain-google-genai 4.4.1 imports `google.genai` only; `google.generativeai` appears only in a docstring. | **[EXECUTED]** grep of the installed package |
| A6 | Medium | The pipeline is duplicated across `app.py` and `streamlit_app.py`, with no config module. | **[INSPECTED]** |

### Error handling and input validation

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| F1 | Medium | No exception handling around `chain.invoke`. API, quota and network errors surface as raw Streamlit tracebacks. A missing index (G2) produces a FAISS C++ error with no guidance. | **[INSPECTED]** + **[EXECUTED]** (G2) |
| F2 | Low | Whitespace-only questions pass `if question:` and are sent to the LLM. | **[INSPECTED]** |

### Tests, CI and README

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| H1 | High | No tests, no test framework, no CI. `app.py` step 7 is a manual smoke check that needs a live paid API. | **[INSPECTED]** |
| R1 | Medium | The README clone URL is a placeholder (`github.com/yourusername/rag-qa-system.git`). | **[INSPECTED]** |
| R2 | High | The README says "Add PDF files to data/ folder", which has no effect: `app.py` reads a hardcoded Windows path, and Docker runs only the UI, which never ingests anything. The index-build step is missing, so the documented Docker flow fails on a fresh clone (G2). | **[INSPECTED]** + **[EXECUTED]** |
| R3 | Low | Commit `0d530d1` "Fix all dependencies in requirements.txt" changed only UI strings in `streamlit_app.py`; `requirements.txt` hasn't changed since `5df7156`. | **[EXECUTED]** `git show --stat` |

### Data and licensing

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| J1 | High (risk) | The 14 PDFs carry the header `DADS304: Visualization … Manipal University Jaipur (MUJ)`. That is university course material redistributed in a public repository, and redistribution rights are unverified. PDF metadata also includes an author username. Recommendation: remove the PDFs from the public repo. Purging them from history needs a force-push, which is the owner's decision. This is not legal advice. | **[EXECUTED]** text extraction |
| J2 | Medium | No `LICENSE` file. | **[INSPECTED]** |
| J3 | OK | **No secrets in history.** A pattern scan of all 4 commits for `AIza…`, `sk-…` and `GOOGLE_API_KEY=<value>` found nothing. `.env` was never committed. | **[EXECUTED]** |

## Assumptions (not verified)

1. **Gemini calls fail today.** Based on Google's shutdown date (D1), not on an API call.
2. **Cause of the Unit-01 corruption.** Every LF in the file is part of a CRLF (6,857 of each), while the healthy PDFs have thousands of lone LFs. That fits a text-mode line-ending conversion. Git's autocrlf is unlikely to be the cause because the file has NUL bytes in its first 8 KB, so git treats it as binary. The corruption probably happened before the commit.
3. **The author's local Docker image probably worked** because a locally built `vector_db/` was copied in through G1.
4. **Image size above 6 GB** is inferred from the venv size (A3). No image was built.
5. **`all-MiniLM-L6-v2` output is L2-normalised**, which would make FAISS `IndexFlatL2` ranking equivalent to cosine ranking. Not checked, because the model could not be downloaded.
6. **GitHub's dependency graph and Dependabot probably cannot parse a UTF-16 `requirements.txt`.** Not tested.

## Ranking: impact × implementation risk

The goal is RAG Reliability Lab. Fixing the legacy chatbot only matters where it blocks measurement or is a security risk.

| Rank | Problem | Impact | Risk to fix | Plan |
|---|---|---|---|---|
| 1 | No way to measure quality: no dataset, metrics, citations or tests (E1, H1, C1, D2) | Very high: the core of the product | Medium | **Milestone 1** |
| 2 | Silent ingestion loss and no ingestion report (B2–B5) | High: invalidates any measurement | Low | **Milestone 1** (lab ingestion) |
| 3 | Secret leaks into images (G1) | Critical security | Very low (rename) | **Milestone 1**, separate commit |
| 4 | Unpinned, UTF-16, CUDA-heavy dependencies (A1–A5) | High: no reproducibility | Low | **Milestone 1** for the lab (pyproject plus lock); legacy file in Milestone 3 |
| 5 | Dead model, broken Docker flow, hardcoded path (D1, G2, B1) | Critical for the legacy app | Medium (needs index build in the image, model config, an API key to verify) | **Milestone 3**: rebuild the Q&A app and Docker on the shared pipeline |
| 6 | Configuration comparisons, LLM providers, replayable LLM runs | High (product) | Medium | **Milestone 2** |
| 7 | Streamlit results inspector, architecture docs, screenshots | Medium (portfolio) | Low | **Milestone 3** |
| 8 | Course PDFs in a public repo (J1), no LICENSE (J2) | Legal/portfolio risk | Needs an owner decision | Flagged; owner to decide |

**Milestone 1, chosen and implemented: a deterministic evaluation core.** It covers a typed package, an original version-controlled dataset, retrieval and answer metrics with explicit definitions, JSON and Markdown reports, a CLI, tests, CI, and a baseline that is reproducible offline. It ships the highest-value capability, can be fully verified in this environment without paid APIs, and creates the measurement harness every later fix gets judged against.

## Appendix: key commands run

```bash
pip install --dry-run --report r.json -r requirements.txt   # resolution, CUDA packages
pip install -r requirements.txt                             # 322 s, EXIT=0
python app.py                                               # FileNotFoundError (B1)
python app_pathpatched.py                                   # 418 pages, 594 chunks, then HF 403
python probe.py                                             # per-file chunks, prompt capture, FAISS error
docker build (FROM scratch probe) with dockerignore vs .dockerignore   # G1
pdftotext / pypdf on each PDF; byte counts of CR/LF          # B3
git log -p --all | grep -E 'AIza…|sk-…|GOOGLE_API_KEY=…'     # J3
```
