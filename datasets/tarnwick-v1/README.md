# Dataset card: tarnwick-v1

A small, version-controlled RAG evaluation set built around the internal knowledge base of a **fictional** company, Tarnwick Systems. Every organisation, product, person and number in it is invented.

## Why a fictional corpus

- **Answers must come from retrieval.** A model cannot know Tarnwick's mileage rate from pre-training, so a correct answer is evidence of grounding, not memory.
- **Redistributable.** The text was written for this repository, so it carries no third-party copyright and no customer data. The course PDFs in `data/` are third-party material (see `PROJECT_AUDIT.md`, J1) and are deliberately **not** used.
- **Controllable difficulty.** Near-duplicate documents, a superseded policy and version history were written in on purpose to produce distractor and conflict cases.

## Composition

| | |
|---|---|
| Documents | 16 Markdown files, about 20 K characters. 12 hold answers; 4 are topical noise (office guide, installation guide, learning budget, on-call handbook) that share vocabulary but answer nothing |
| Questions | 56: 47 answerable, 9 unanswerable |
| Evidence spans | 55 across the 47 answerable questions (8 questions need two spans) |
| Chunks at 1000/200 | 29 (55 at 500/100, 100 at 300/50) |

| Category | n | What it tests |
|---|---|---|
| `factual` | 12 | A direct question answered by one sentence |
| `multi_passage` | 8 | Needs two evidence spans, often from different documents (e.g. warranty in the FAQ + battery life in the datasheet) |
| `unanswerable` | 8 | Plausible near-misses: the topic exists (on-call, KX-220 limits, Enterprise plan) but the asked fact does not |
| `distractor` | 8 | A similar but wrong passage exists (KX-200 vs KX-220, employee vs contractor onboarding, daily vs monthly backups) |
| `conflicting` | 7 | The 2024 policy or older gateway release says one thing; the superseding 2026 policy or 3.4.0 release says another. `stale_facts` hold the outdated values |
| `poorly_phrased` | 7 | Typos, fragments, lowercase (`wat happens if i lose a recipt`); one is also unanswerable |
| `attribution` | 6 | The question names its source ("According to the 2024 edition…"); the citation must point to that document, not its near-duplicate |

## Schema

See the docstring of `src/rag_reliability_lab/dataset.py`. In short, each line of `questions.jsonl` has `id`, `question`, `category`, `answerable`, span-level `evidence` (`doc` + exact `quote`), `reference_answer`, `required_facts` with aliases, and optional `stale_facts`, `absent_terms` and `notes`.

## Guarantees enforced by `rag-lab validate-dataset`

- Every evidence quote occurs **exactly once** in its document, so its offsets are unambiguous.
- Every non-derived required fact appears in that question's evidence quotes, so the gold answer is grounded.
- Every stale fact actually occurs somewhere in the corpus.
- Every `absent_terms` phrase of an unanswerable question occurs **nowhere** in the corpus. This is what shows the question really is unanswerable, rather than just asserting it.
- Unknown keys, duplicate ids, unknown documents and corpus extraction problems are errors.

## Known limitations and biases

- **Small.** At n = 56, 95% intervals on rates are roughly ±13 percentage points. Use it to find failure modes and catch regressions, not to rank models.
- **Single author, and the same author wrote documents and questions.** Question wording overlaps the documents' vocabulary more than real user questions would, which favours lexical retrieval (BM25). The `poorly_phrased` category partly offsets this.
- **Easy retrieval at large chunk sizes.** With 29 chunks at 1000/200, top-3 already covers about 10% of the corpus, so retrieval is close to ceiling. Smaller chunks, or a larger noise corpus, make retrieval differences visible.
- **English only, Markdown only.** No PDF or OCR artefacts, tables or long documents.
- **One question needs world knowledge:** `p03` relies on knowing Portugal is an EU country. This is flagged in its `notes`.
- **Alias lists are finite.** A correct paraphrase can be scored as missing. Extend `any_of` when an audit finds a false negative, and bump the version.

## Versioning

Any change to `corpus/` or `questions.jsonl` must bump `version` in `dataset.toml`. Reports record `questions_sha256` and a corpus fingerprint, so results from different dataset versions cannot be silently compared.

## License

Not yet chosen by the repository owner. The content is original to this repository.
