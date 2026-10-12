# Scoring definitions

Every number in a RAG Reliability Lab report comes from a deterministic rule on this page. No LLM judge is involved. Retrieval quality and answer quality are scored separately, so a failure can be traced to the stage that caused it.

## 1. Evidence and relevance

The dataset annotates evidence as **spans**: an exact quote inside a document, located by character offsets. It does not use chunk ids. The same labels therefore work for any chunk size.

A retrieved chunk **covers** an evidence span when both come from the same document and the chunk contains at least `τ` of the span's characters:

```
coverage(chunk, span) = |chars(chunk) ∩ chars(span)| / |chars(span)|
covers(chunk, span)   = coverage ≥ τ            (τ = evaluation.evidence_overlap, default 0.5)
```

A chunk is **relevant** to a question if it covers at least one of the question's evidence spans. With τ = 0.5, a span split across two chunks still counts once either chunk holds half of it.

Chunk offsets are exact by construction (see `chunking.py`), and `text[start:end] == chunk` is verified for every chunk.

## 2. Retrieval metrics

These are computed for **answerable questions only**. Unanswerable questions have no evidence and are scored on answer behaviour alone. Let `E` be the question's evidence spans and `R_k` the top-k retrieved chunks.

| Metric | Definition | Notes |
|---|---|---|
| `hit@k` | 1 if any chunk in `R_k` is relevant | "Did we find anything useful?" |
| `complete@k` | 1 if every span in `E` is covered by `R_k` | Strict; matters for multi-passage questions |
| `recall@k` | covered spans / `|E|` | Span-level recall |
| `precision@k` | relevant chunks in `R_k` / **k** | Denominator is k even if fewer chunks were returned. Bounded by (chunks that can cover the evidence) / k, so with one evidence span and k = 3 it is usually at most 1/3, or 2/3 when overlap puts the span in two chunks. Compare precision between configurations, not against 1.0. |
| `rr@k` | 1 / rank of the first relevant chunk, else 0 | Mean over questions = **MRR@k** |

Reports show these at the system's `top_k` (with 95% CIs) and at diagnostic cutoffs (default 1, 3, 5, 10). The generator only ever sees the first `top_k` chunks; deeper cutoffs answer the question "would a larger k help?"

## 3. Answer metrics

### Fact matching

Each answerable question lists **required facts**. Each fact has aliases (`any_of`), for example `["2-year", "2 years", "two years"]`. A fact is present when any alias occurs as a **contiguous token sequence** in the answer. Tokenisation (`text.match_tokens`):

- NFKC normalisation, casefolding, and unified quotes and dashes (U+2212 minus becomes `-`);
- numbers stay whole (`0.34`, `1,500`, `10:00`), and a leading minus is kept (`-20`) except inside ranges (`20-60` gives `20`, `60`);
- `€ $ £ % °` are tokens; other punctuation is ignored.

So `"14 days"` matches "within 14 days" but not "114 days" or "14 calendar days". Paraphrases need their own alias. The validator requires every non-derived fact to appear in its evidence quotes.

### Per-question answer fields

| Field | Definition |
|---|---|
| `abstained` | The answer matches an abstention pattern (below) |
| `answered` | Not a bare abstention: no abstention pattern, **or** at least one required fact stated (a hedged answer) |
| `fact_coverage` | Required facts present / required facts |
| `answer_correct` | Answerable: `fact_coverage == 1`. Unanswerable: `abstained` |
| `stale_facts_present` | Outdated values (`stale_facts`) found in the answer |
| `stale_only` | A stale value is present and **no** required fact is (the system gave the outdated answer) |
| `citations` | Valid `[P<n>]` labels, where n is between 1 and the number of passages the generator received |
| `invalid_citations` | Labels outside that range (invented passages) |
| `citation_precision` | Cited passages that cover any evidence span / cited passages |
| `citation_evidence_recall` | Evidence spans covered by at least one cited passage / evidence spans |
| `attribution_correct` | Answerable and answered: ≥ 1 valid citation, no invalid ones, and `citation_evidence_recall == 1` |
| `supported_fact_rate` | Matched required facts that also appear in the **cited** passages' text / matched facts. A grounding check: a fact the answer states but none of its citations contain is unsupported. |

Stating both the stale and the current value counts as acknowledging the conflict, and the answer can still be correct. Reports also count how often a stale value is mentioned at all.

### Citation parsing

`[P1]`, `[P1][P3]`, `[P1, P3]`, `[P2; P4]` and `[ p 5 ]` are recognised. A bracket group counts as a citation group only if at least one part parses as `P<n>`, so ordinary text like `[sic]` is ignored. Non-label parts of a citation group are reported as `malformed_citations`.

### Abstention patterns

Checked in order on the normalised answer. The first pattern that matches is recorded in the report for auditability.

1. `refusal_text`: the exact sentence the prompt asks for, *"I could not find the answer in the provided documents."*
2. `could_not_find`: could not / couldn't / cannot / can't find
3. `not_in_sources`: "not (mentioned|stated|specified|provided|included|covered|found|available)" followed within 40 characters by documents / passages / context / sources
4. `sources_do_not`: documents / passages / context / sources followed within 30 characters by "do not / does not / did not (contain|mention|state|specify|provide|include|say|cover)"
5. `no_information`: "no information", "no mention", "no details"

Rule-based detection can be fooled. An answer such as "No details are needed for receipts under EUR 40" trips pattern 5. Every abstention shows which pattern fired, so these cases can be audited.

## 4. Diagnosis: the first stage that failed

Each question gets exactly one label, assigned in this order:

| Label | Condition |
|---|---|
| `error` | Retrieval or generation raised an exception (timeout, API error, …) |
| `pass` | Answerable: answer correct **and** attribution correct. Unanswerable: abstained. |
| `citation_error` | Answer correct, attribution wrong |
| `retrieval_miss` | Answer incorrect and the evidence was not completely retrieved at `top_k` |
| `stale_answer` | Evidence retrieved, answer gives only the outdated value |
| `false_abstention` | Evidence retrieved, but the system abstained |
| `generation_miss` | Evidence retrieved, answered, but required facts missing |
| `hallucinated_answer` | Unanswerable question, but the system answered |

In retrieval-only runs (`provider = "none"`) answerable questions are `pass` or `retrieval_miss` based on `complete@k`; unanswerable questions are `not_applicable` and are left out of pass rates.

The summary metric "false abstention" counts **every** abstention on an answerable question, whatever the cause. The diagnosis `false_abstention` is assigned only when the evidence *was* retrieved. An abstention after a retrieval miss is diagnosed `retrieval_miss`.

## 5. Statistics and honesty rules

- **Errors count as failures.** A question whose retrieval raised an error contributes zeros to retrieval metrics, and any error makes the answer incorrect. Errors are listed in the report and the CLI exits with code 2.
- **Proportions** (hit, complete, correct, abstention rates) carry **Wilson score 95% intervals**, which stay valid at small n and near 0 or 100%.
- **Means** (recall, precision, MRR, fact coverage) carry **percentile bootstrap 95% intervals** (10,000 resamples, seed 0). The sampler uses only `random.random()`, the one generator method Python guarantees to reproduce across versions.
- Per-category rows have small n and no intervals. Use them for diagnosis, not as estimates.
- **Comparing two configurations** on this dataset requires a paired test on per-question outcomes (planned for Milestone 2). Two overlapping intervals from independent summaries are not evidence of a difference.
- Parameters of the deterministic baselines (BM25 k1/b, extractive `min_overlap`) were fixed **a priori**, not tuned on the evaluation set.

## 6. What this scoring does not measure

- Semantic equivalence beyond the listed aliases. A correct paraphrase without an alias is scored as missing. Alias lists are part of the dataset and should grow when audits find false negatives.
- Extra wrong claims, unless they are listed as `stale_facts`. An answer can contain every required fact plus an irrelevant or wrong sentence and still be scored correct. The supported-fact rate and citation precision partly expose this.
- Fluency, tone and helpfulness.
- Anything an LLM judge would add. A model-based judge may be added later as a **secondary**, clearly labelled signal, never as the only source of truth.
