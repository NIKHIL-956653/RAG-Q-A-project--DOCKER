# RAG evaluation report: baseline-offline

Legacy chunking (1000/200, k=3) with BM25 retrieval and the extractive answerer. Deterministic.

- **Dataset:** tarnwick-v1 v1.0.0. 56 of 56 questions evaluated (47 answerable, 9 unanswerable), 16 documents
- **Chunking:** recursive_character 1000/200 -> 29 chunks
- **Retriever:** method=bm25, k1=1.2, b=0.75, tokenizer=retrieval_tokens/v1; top_k=3
- **Generator:** provider=extractive, max_sentences=2, min_overlap=0.5
- **Config sha256:** `ef98d27a7decf659` · **package** 0.1.0
- **Errors:** 0 (errors count as failures)

> **Read this first.** This is a small, hand-built dataset. Every rate comes with a 95% confidence interval. Differences smaller than the interval width are not evidence that one configuration is better. Retrieval and answer quality are scored separately and deterministically (no LLM judge). See `docs/SCORING.md`.

## Retrieval quality (answerable questions, k=3)

| Metric | Value | 95% CI |
|---|---|---|
| Hit@3 (any evidence retrieved) | 97.9% (46/47) | [88.9%, 99.6%] |
| Complete@3 (all evidence retrieved) | 97.9% (46/47) | [88.9%, 99.6%] |
| Recall@3 | 0.979 (n=47) | [0.936, 1.000] |
| Precision@3 | 0.383 (n=47) | [0.340, 0.426] |
| MRR@3 | 0.876 (n=47) | [0.801, 0.943] |

| Cutoff | Hit | Complete | Recall | Precision | MRR |
|---|---|---|---|---|---|
| @1 | 78.7% | 68.1% | 0.734 | 0.787 | 0.787 |
| @3 | 97.9% | 97.9% | 0.979 | 0.383 | 0.876 |
| @5 | 97.9% | 97.9% | 0.979 | 0.230 | 0.876 |
| @10 | 97.9% | 97.9% | 0.979 | 0.115 | 0.876 |

## Answer quality

| Metric | Value | 95% CI |
|---|---|---|
| **Fully correct** (answer + attribution, or correct abstention) | 57.1% (32/56) | [44.1%, 69.2%] |
| Answerable: answer correct (all required facts) | 55.3% (26/47) | [41.2%, 68.6%] |
| Answerable: fact coverage | 0.585 (n=47) | [0.447, 0.723] |
| Answerable: false abstention | 29.8% (14/47) | [18.7%, 44.0%] |
| Answerable: attribution correct (when answered) | 97.0% (32/33) | [84.7%, 99.5%] |
| Unanswerable: correct abstention | 66.7% (6/9) | [35.4%, 87.9%] |
| Unanswerable: hallucinated answer | 33.3% (3/9) | [12.1%, 64.6%] |

Citation precision (answers with citations): 0.758 (n=33) · Supported-fact rate: 1.000 (n=29) · Stale-only answers: 0/7 · Stale value mentioned (alongside or instead of the current one): 6/7

## Diagnosis

| Diagnosis | Count |
|---|---|
| pass | 32 |
| retrieval_miss | 1 |
| false_abstention | 13 |
| generation_miss | 7 |
| hallucinated_answer | 3 |

## By category

| Category | n | Hit@k | Recall@k | Answer correct | Pass |
|---|---|---|---|---|---|
| factual | 12 | 100.0% | 1.000 | 100.0% | 100.0% |
| multi_passage | 8 | 100.0% | 1.000 | 0.0% | 0.0% |
| unanswerable | 8 | n/a | n/a | 62.5% | 62.5% |
| distractor | 8 | 100.0% | 1.000 | 37.5% | 37.5% |
| conflicting | 7 | 100.0% | 1.000 | 85.7% | 85.7% |
| poorly_phrased | 7 | 83.3% | 0.833 | 28.6% | 28.6% |
| attribution | 6 | 100.0% | 1.000 | 66.7% | 66.7% |

Per-category n is small; treat these rows as diagnostic, not as estimates.

## Failed questions (24)

### m01 · multi_passage · `false_abstention`

**Q:** What warranty does the KX-200 have, and how long does its battery typically last?

**Expected evidence:**
- [retrieved] `support/support-faq.md`: "The KX-200 sensor has a 2-year limited warranty."
- [retrieved] `products/kx-200-datasheet.md`: "Typical battery life is 18 months at the default 15-minute sampling interval."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `support/support-faq.md::c0000` | 11.399 | 0 |
| 2 | `products/kx-200-datasheet.md::c0000` | 8.767 | 1 |
| 3 | `products/kx-installation-guide.md::c0000` | 6.342 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: warranty, battery; abstention: refusal_text_

**Reference:** A 2-year limited warranty (batteries excluded); typical battery life is 18 months at the default 15-minute interval.

### m02 · multi_passage · `false_abstention`

**Q:** What is the KX-220's ingress protection rating, and how long is its warranty?

**Expected evidence:**
- [retrieved] `products/kx-220-datasheet.md`: "Ingress protection: IP67 (dust tight, protected against temporary immersion)"
- [retrieved] `support/support-faq.md`: "The KX-220 sensor has a 3-year limited warranty."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `products/kx-220-datasheet.md::c0000` | 9.005 | 0 |
| 2 | `support/support-faq.md::c0000` | 8.953 | 1 |
| 3 | `products/kx-200-datasheet.md::c0000` | 6.140 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: ip, warranty; abstention: refusal_text_

**Reference:** IP67, with a 3-year limited warranty.

### m03 · multi_passage · `generation_miss`

**Q:** How long is the probation period for a new permanent employee, and how many days per week can they work remotely once it is over?

**Expected evidence:**
- [retrieved] `handbook/employee-onboarding.md`: "The probation period for permanent employees is six months."
- [retrieved] `handbook/remote-work-policy.md`: "All permanent employees who have completed their probation period may work remotely up to four days per week."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/employee-onboarding.md::c0000` | 15.391 | 0 |
| 2 | `handbook/employee-onboarding.md::c0001` | 13.981 | 0 |
| 3 | `handbook/remote-work-policy.md::c0000` | 13.527 | 1 |

**Answer:** All permanent employees who have completed their probation period may work remotely up to four days per week. [P3]

_missing facts: probation; citations: [3] invalid: []_

**Reference:** Probation lasts six months; after it, up to four days per week remotely.

### m04 · multi_passage · `false_abstention`

**Q:** What are the support hours for Standard plan customers, and how quickly do they get a first response?

**Expected evidence:**
- [retrieved] `support/support-faq.md`: "Customers on the Standard plan can reach support Monday to Friday, 08:00 to 18:00 Central European Time, excluding public holidays."
- [retrieved] `support/support-faq.md`: "Standard plan tickets receive a first response within one business day."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `support/support-faq.md::c0000` | 17.625 | 0, 1 |
| 2 | `engineering/data-retention-standard.md::c0000` | 8.354 | - |
| 3 | `engineering/incident-response-runbook.md::c0000` | 6.271 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: hours, response; abstention: refusal_text_

**Reference:** Monday to Friday 08:00-18:00 CET (excluding public holidays); first response within one business day.

### m05 · multi_passage · `false_abstention`

**Q:** By when must a new contractor finish security training, and who has to request an extension of their account?

**Expected evidence:**
- [retrieved] `handbook/contractor-onboarding.md`: "Contractors must complete the information security awareness course within seven days of their account being activated."
- [retrieved] `handbook/contractor-onboarding.md`: "An extension must be requested by the sponsor at least five working days before the expiry date."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/contractor-onboarding.md::c0000` | 11.770 | 1 |
| 2 | `handbook/contractor-onboarding.md::c0001` | 10.517 | 0 |
| 3 | `handbook/employee-onboarding.md::c0000` | 5.319 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: training, who; abstention: refusal_text_

**Reference:** Within seven days of account activation; the sponsor requests extensions at least five working days before expiry.

### m06 · multi_passage · `generation_miss`

**Q:** How long are daily database backups kept, and is data deleted on a customer's request restored from those backups?

**Expected evidence:**
- [retrieved] `engineering/data-retention-standard.md`: "Database backups are taken daily and retained for 35 days."
- [retrieved] `engineering/data-retention-standard.md`: "Deleted data can remain in database backups until those backups expire under the backup retention period above; it is not restored into production."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `engineering/data-retention-standard.md::c0000` | 27.126 | 0, 1 |
| 2 | `support/support-faq.md::c0001` | 4.560 | - |
| 3 | `handbook/remote-work-policy.md::c0001` | 3.492 | - |

**Answer:** Deleted data can remain in database backups until those backups expire under the backup retention period above; it is not restored into production. [P1]

_missing facts: backup_days; citations: [1] invalid: []_

**Reference:** Daily backups are kept for 35 days; deleted data may remain in backups until they expire but is not restored into production.

### m07 · multi_passage · `false_abstention`

**Q:** When does a SEV-1 page escalate to the secondary on-call engineer, and who approves a SEV-1 postmortem?

**Expected evidence:**
- [retrieved] `engineering/incident-response-runbook.md`: "If the page is not acknowledged within 5 minutes, it escalates to the secondary on-call engineer."
- [retrieved] `engineering/incident-response-runbook.md`: "SEV-1 postmortems are approved by the VP of Engineering"

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `engineering/incident-response-runbook.md::c0000` | 18.232 | 0 |
| 2 | `engineering/incident-response-runbook.md::c0001` | 17.866 | 1 |
| 3 | `engineering/on-call-handbook.md::c0000` | 9.201 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: escalation, approver; abstention: refusal_text_

**Reference:** If not acknowledged within 5 minutes; the VP of Engineering approves SEV-1 postmortems.

### m08 · multi_passage · `generation_miss`

**Q:** What sampling interval range does the KX-220 support, and what is its typical battery life?

**Expected evidence:**
- [retrieved] `products/kx-220-datasheet.md`: "The sampling interval can be configured between 1 and 120 minutes."
- [retrieved] `products/kx-220-datasheet.md`: "Typical battery life is 36 months at the default 15-minute sampling interval."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `products/kx-220-datasheet.md::c0000` | 19.477 | 0, 1 |
| 2 | `products/kx-200-datasheet.md::c0000` | 18.421 | - |
| 3 | `products/kx-220-datasheet.md::c0001` | 12.196 | 0 |

**Answer:** Typical battery life is 36 months at the default 15-minute sampling interval. [P1] Typical battery life is 18 months at the default 15-minute sampling interval. [P2]

_missing facts: interval; citations: [1, 2] invalid: []_

**Reference:** 1 to 120 minutes; typical battery life is 36 months at the default 15-minute interval.

### u01 · unanswerable · `hallucinated_answer`

**Q:** What is Tarnwick's parental leave policy?

**Expected:** abstention (not answerable from the corpus).

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/travel-expense-policy-2024.md::c0000` | 3.931 | - |
| 2 | `handbook/travel-expense-policy-2026.md::c0000` | 3.427 | - |
| 3 | `handbook/learning-budget.md::c0000` | 1.729 | - |

**Answer:** This policy applies to all Tarnwick Systems employees who travel on company business. [P1] This policy applies to all Tarnwick Systems employees who travel on company business, including interns. [P2]

_citations: [1, 2] invalid: []_

### u03 · unanswerable · `hallucinated_answer`

**Q:** Who is the CEO of Tarnwick Systems?

**Expected:** abstention (not answerable from the corpus).

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/travel-expense-policy-2024.md::c0000` | 2.642 | - |
| 2 | `handbook/travel-expense-policy-2026.md::c0000` | 2.526 | - |
| 3 | `handbook/employee-onboarding.md::c0001` | 2.217 | - |

**Answer:** This policy applies to all Tarnwick Systems employees who travel on company business. [P1] This policy applies to all Tarnwick Systems employees who travel on company business, including interns. [P2]

_citations: [1, 2] invalid: []_

### u05 · unanswerable · `hallucinated_answer`

**Q:** How many vacation days do employees get per year?

**Expected:** abstention (not answerable from the corpus).

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/learning-budget.md::c0000` | 4.137 | - |
| 2 | `handbook/remote-work-policy.md::c0001` | 3.953 | - |
| 3 | `handbook/travel-expense-policy-2026.md::c0001` | 3.433 | - |

**Answer:** Employees may spend up to five working days per year on learning activities during working hours, agreed in advance with their manager. [P1] Employees may work from another EU or EEA country for up to 30 working days per calendar year. [P2]

_citations: [1, 2] invalid: []_

### d01 · distractor · `false_abstention`

**Q:** What is the operating temperature range of the KX-220?

**Expected evidence:**
- [retrieved] `products/kx-220-datasheet.md`: "Operating temperature: -40 °C to 70 °C"

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `products/kx-220-datasheet.md::c0000` | 12.543 | 0 |
| 2 | `products/kx-200-datasheet.md::c0000` | 10.315 | - |
| 3 | `products/kx-installation-guide.md::c0000` | 5.965 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: min, max; abstention: refusal_text_

**Reference:** -40 °C to 70 °C.

### d02 · distractor · `false_abstention`

**Q:** What ingress protection rating does the KX-200 have?

**Expected evidence:**
- [retrieved] `products/kx-200-datasheet.md`: "Ingress protection: IP65 (dust tight, protected against water jets)"

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `products/kx-200-datasheet.md::c0000` | 9.199 | 0 |
| 2 | `products/kx-installation-guide.md::c0000` | 6.342 | - |
| 3 | `products/kx-220-datasheet.md::c0000` | 6.303 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: ip; abstention: refusal_text_

**Reference:** IP65.

### d03 · distractor · `generation_miss`

**Q:** Do contractors receive a company laptop?

**Expected evidence:**
- [retrieved] `handbook/contractor-onboarding.md`: "Contractors do not receive a company laptop. They work through a virtual desktop account"

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/contractor-onboarding.md::c0000` | 8.881 | 0 |
| 2 | `support/support-faq.md::c0000` | 3.159 | - |
| 3 | `handbook/contractor-onboarding.md::c0001` | 3.141 | - |

**Answer:** Contractors do not receive a company laptop. [P1]

_missing facts: alternative; citations: [1] invalid: []_

**Reference:** No. Contractors work through a virtual desktop account.

### d05 · distractor · `false_abstention`

**Q:** How long is the monthly database backup kept?

**Expected evidence:**
- [retrieved] `engineering/data-retention-standard.md`: "In addition, one backup per month is kept for 12 months."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `engineering/data-retention-standard.md::c0000` | 12.922 | 0 |
| 2 | `security/access-review-procedure.md::c0000` | 2.056 | - |
| 3 | `handbook/travel-expense-policy-2026.md::c0001` | 1.948 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: period; abstention: refusal_text_

**Reference:** 12 months.

### d08 · distractor · `generation_miss`

**Q:** What is the storage temperature range of the KX-200?

**Expected evidence:**
- [retrieved] `products/kx-200-datasheet.md`: "Storage temperature: -30 °C to 70 °C"

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `products/kx-200-datasheet.md::c0000` | 12.884 | 0 |
| 2 | `products/kx-220-datasheet.md::c0000` | 9.351 | - |
| 3 | `products/kx-installation-guide.md::c0000` | 6.342 | - |

**Answer:** The KX-200 is a battery-powered indoor and sheltered-outdoor sensor that measures particulate matter (PM2.5 and PM10), temperature and relative humidity. [P1]

_missing facts: min, max; citations: [1] invalid: []_

**Reference:** -30 °C to 70 °C.

### c01 · conflicting · `false_abstention`

**Q:** What is the mileage reimbursement rate for business trips in a private car?

**Expected evidence:**
- [retrieved] `handbook/travel-expense-policy-2026.md`: "Employees who use a private car for business travel are reimbursed at EUR 0.34 per kilometre"

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/travel-expense-policy-2024.md::c0001` | 10.901 | - |
| 2 | `handbook/travel-expense-policy-2026.md::c0001` | 10.124 | 0 |
| 3 | `handbook/travel-expense-policy-2026.md::c0000` | 5.052 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: rate; abstention: refusal_text_

**Reference:** EUR 0.34 per kilometre under the 2026 policy (the superseded 2024 rate was EUR 0.30).

### p01 · poorly_phrased · `retrieval_miss`

**Q:** wat happens if i lose a recipt

**Expected evidence:**
- [**MISSING**] `handbook/travel-expense-policy-2026.md`: "If a receipt is lost, the employee must complete a Missing Receipt Declaration (form FIN-07) within 10 working days of the expense."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `engineering/on-call-handbook.md::c0000` | 2.678 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: form; abstention: refusal_text_

**Reference:** Complete a Missing Receipt Declaration (form FIN-07) within 10 working days of the expense.

### p02 · poorly_phrased · `false_abstention`

**Q:** postmortem deadline sev1?

**Expected evidence:**
- [retrieved] `engineering/incident-response-runbook.md`: "The postmortem must be published within five working days of the incident being resolved."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `engineering/incident-response-runbook.md::c0001` | 4.600 | 0 |
| 2 | `security/access-review-procedure.md::c0000` | 2.867 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: deadline; abstention: refusal_text_

**Reference:** Within five working days of the incident being resolved.

### p03 · poorly_phrased · `generation_miss`

**Q:** can i work remote from portugal

**Expected evidence:**
- [retrieved] `handbook/remote-work-policy.md`: "Employees may work from another EU or EEA country for up to 30 working days per calendar year."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/remote-work-policy.md::c0000` | 4.684 | - |
| 2 | `handbook/remote-work-policy.md::c0001` | 4.434 | 0 |
| 3 | `handbook/contractor-onboarding.md::c0000` | 2.156 | - |

**Answer:** Remote Work Policy [P1] Remote work must take place on a private network or over the company VPN. [P2]

_missing facts: limit; citations: [1, 2] invalid: []_

**Reference:** Yes, Portugal is in the EU: up to 30 working days per calendar year, approved by People Operations at least 14 days in advance.

### p04 · poorly_phrased · `false_abstention`

**Q:** how long r app logs kept

**Expected evidence:**
- [retrieved] `engineering/data-retention-standard.md`: "Application logs are retained for 90 days and then deleted automatically."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `engineering/data-retention-standard.md::c0000` | 9.127 | 0 |
| 2 | `facilities/office-guide.md::c0000` | 2.797 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: period; abstention: refusal_text_

**Reference:** 90 days, then deleted automatically.

### p07 · poorly_phrased · `generation_miss`

**Q:** core hours remote?

**Expected evidence:**
- [retrieved] `handbook/remote-work-policy.md`: "Remote employees must be reachable between 10:00 and 15:00 Central European Time on working days."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/remote-work-policy.md::c0000` | 6.275 | 0 |
| 2 | `handbook/contractor-onboarding.md::c0001` | 4.523 | - |
| 3 | `handbook/remote-work-policy.md::c0001` | 2.631 | - |

**Answer:** Core collaboration hours [P1] The core collaboration hours that apply to employees do not apply to contractors. [P2]

_missing facts: hours; citations: [1, 2] invalid: []_

**Reference:** 10:00 to 15:00 Central European Time on working days.

### a01 · attribution · `false_abstention`

**Q:** According to the 2024 edition of the travel policy, what was the mileage rate for a private car?

**Expected evidence:**
- [retrieved] `handbook/travel-expense-policy-2024.md`: "Employees who use a private car for business travel are reimbursed at EUR 0.30 per kilometre."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `handbook/travel-expense-policy-2024.md::c0000` | 12.344 | - |
| 2 | `handbook/travel-expense-policy-2026.md::c0000` | 11.572 | - |
| 3 | `handbook/travel-expense-policy-2024.md::c0001` | 7.800 | 0 |

**Answer:** I could not find the answer in the provided documents.

_missing facts: rate; abstention: refusal_text_

**Reference:** EUR 0.30 per kilometre (2024 edition, now superseded).

### a06 · attribution · `false_abstention`

**Q:** According to the Relay Gateway release notes, support for which sensor was removed in version 3.4.0?

**Expected evidence:**
- [retrieved] `products/relay-gateway-release-notes.md`: "Support for the discontinued KX-100 sensor has been removed."

| Rank | Chunk | Score | Covers evidence |
|---|---|---|---|
| 1 | `products/relay-gateway-release-notes.md::c0000` | 23.101 | 0 |
| 2 | `products/relay-gateway-release-notes.md::c0001` | 16.898 | - |
| 3 | `support/support-faq.md::c0000` | 10.021 | - |

**Answer:** I could not find the answer in the provided documents.

_missing facts: model; abstention: refusal_text_

**Reference:** The discontinued KX-100.

## Ingestion

16 files -> 16 documents -> 29 chunks (chunk chars min/median/max: 190/840/998). Issues: 0.


## Latency and environment (not deterministic)

Retrieval p50 0.04 ms, p95 0.08 ms · Generation p50 0.39 ms, p95 0.65 ms · index build 2.4 ms · total 0.10 s · Python 3.11.17 on Linux-6.18.44-fc-v114-x86_64-with-glibc2.39 · pypdf 6.20.0 · git `d314bf5+dirty`

## Reproduce

```bash
pip install -e .
rag-lab eval --config configs/baseline-offline.toml --out <dir>
```

Expected hashes: config `ef98d27a7decf6595a52d14b8042187ffc88bb6438957cfecf160de0754ada47`, questions `3dabc1026d0bef94d2531552a6acb2627c433539e7e0a0b116b4c8ffa36782bd`, corpus `1810bca5a580cfa3ab31f6c0e9ed4ca22f172e2e626facc7a0d8745416891b3e`.
