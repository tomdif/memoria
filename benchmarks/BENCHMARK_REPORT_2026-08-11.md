# Memoria benchmark report — 2026-08-11

## Bottom line

Memoria's session retriever is strong on LongMemEval-S, and the new grouped
window retriever substantially improves LoCoMo.

- On the official LongMemEval-S retrieval scope, Memoria reaches 94.3%
  Recall-All@5 and 97.1% Recall-All@10. On the identical rows, that is +20.1
  percentage points over flat BM25 and +8.9 points over MiniLM dense retrieval
  at k=5.
- On the four non-adversarial LoCoMo categories with evidence annotations, the
  windowed profile reaches 78.3% Recall-All@5 and 87.5% Recall-All@10. This is
  +27.4 points over the legacy Memoria path and +16.1 points over flat BM25 at
  k=5.
- On the untouched PerLTQA English v2 external holdout, frozen Memoria reaches
  84.4% strict Recall@5 across all 8,593 rows and 85.4% after excluding 102 gold
  IDs absent from the supplied memory banks. That is +3.9 points over windowed
  BM25 and +12.5 points over windowed MiniLM on the identical rendering.
- These are retrieval results, not end-to-end memory-agent or question-answering
  scores. The runner bypasses Memoria's extraction and graph-ingestion path.

This is a useful engineering result, but not a publishable claim that “Memoria
solves long-term memory.” The windowed design was implemented after inspecting
the public LoCoMo result and category failures. It therefore needs confirmation
on an untouched conversational-memory dataset.

## What was run

The benchmark set was selected from official, peer-reviewed or current public
harnesses that are widely used for long-term agent memory:

| Benchmark | Status in this run | Reason |
|---|---|---|
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) | Full session retrieval | Directly supports session-level retrieval labels and official retrieval metrics. |
| [LoCoMo](https://github.com/snap-research/locomo) (ACL 2024) | Full session-evidence diagnostic | Provides evidence dialog IDs, but its official headline task is generated-answer F1 rather than this session metric. |
| [PerLTQA](https://github.com/Elvin-Yiming-Du/PerLTQA) (SIGHAN/ACL 2024) | Full updated English v2 memory retrieval | Provides native reference-memory IDs and officially uses Recall@K. This dataset was not inspected before the retriever was frozen. |
| [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench) (ICLR 2026) | Protocol audited; no score | Scores an answering agent across accurate retrieval, test-time learning, long-range understanding, and conflict resolution. It requires a generator and, for some subsets, an LLM judge. |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) (ICLR 2026) | Protocol audited; no score | Its 2,000 questions over 128K–10M-token conversations evaluate ten abilities through generated answers, not a standalone retriever. |
| [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) (2026) | Adapter requirements audited; no score | The fixed protocol consumes multimodal web trajectories, calls a fixed reader, and jointly evaluates accuracy and latency. The largest haystacks reach 115M tokens. |
| [MemoryArena](https://github.com/ZexueHe/MemoryArena) (ICML 2026) | Environment requirements audited; no score | Requires complete agents, interactive shopping/search/travel/reasoning environments, and configured model/API endpoints. |

No paid API was called. Assigning a retrieval-only proxy to the last four
benchmarks would create numbers that look official but are not comparable to
their leaderboards.

## LongMemEval-S

### Protocol

- Dataset: the current cleaned 500-row LongMemEval-S file.
- Index unit: one session; primary text contains user turns, while Memoria's
  balanced dual-pass reranker can also inspect the user+assistant rendering.
- Scored scope: 419 rows. This mirrors the official retrieval harness, which
  excludes 30 abstention rows and 51 non-abstention rows without a user-side
  `has_answer` target.
- Metrics: the official binary `recall_any`, binary `recall_all`, and
  `ndcg_any` implementation at k=5 and k=10. `Recall-All@k` is 1 only when
  every annotated evidence session occurs in the top k.
- Three rows contain six evidence sessions and therefore cannot pass
  Recall-All@5.

### Results

| Retriever | Recall-Any@5 | Recall-All@5 | NDCG-Any@5 | Recall-Any@10 | Recall-All@10 | NDCG-Any@10 | q/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Memoria balanced** | **98.1%** | **94.3%** | **0.9414** | **99.0%** | **97.1%** | **0.9478** | 3.5 |
| MiniLM dense | 95.9% | 85.4% | 0.8615 | 98.1% | 93.8% | 0.8785 | 6.7 |
| Flat BM25 | 88.8% | 74.2% | 0.7722 | 93.1% | 82.6% | 0.7962 | 346.9 |

[The LongMemEval paper](https://openreview.net/forum?id=wIonk5yTDq)'s original
K=V session-index baselines reported
Recall-All@5 of 63.4% for BM25, 72.3% for Contriever, and 72.0% for Stella V5
1.5B. Those values contextualize the result but are not strict controls because
the official dataset was cleaned again in September 2025. The two local
baselines above are the valid apples-to-apples comparison.

### Memoria by ability

| Ability | n | Recall-All@5 | Recall-All@10 | NDCG-Any@10 |
|---|---:|---:|---:|---:|
| Knowledge update | 72 | 100.0% | 100.0% | 0.9888 |
| Multi-session | 121 | 94.2% | 97.5% | 0.9588 |
| Single-session assistant | 5 | 100.0% | 100.0% | 1.0000 |
| Single-session preference | 30 | 96.7% | 96.7% | 0.9087 |
| Single-session user | 64 | 100.0% | 100.0% | 0.9942 |
| Temporal reasoning | 127 | 87.4% | 93.7% | 0.8978 |

Temporal reasoning accounts for most of the remaining misses.

## LoCoMo

### Protocol

- Dataset: all 10 released conversations and all 1,986 QA rows.
- Index unit for the new Memoria profile: overlapping four-turn windows with a
  two-turn stride. Each window includes the session timestamp, speaker labels,
  dialogue text, and released image captions. Results collapse back to unique
  parent sessions before scoring.
- Candidate ranking: union the dense and BM25 candidates, combine their ranks at
  parent-session level with reciprocal-rank fusion, rerank the best 30 complete
  windows, and add the cross-encoder only as a bounded promotion signal.
- Baselines and the legacy Memoria profile retain whole-session indexing.
- Gold target: the session prefix in each annotated evidence dialog ID, such
  as `D5:12` becoming `session_5`.
- Four rows lack evidence annotations and are excluded from retrieval scoring.
- The core table covers categories 1–4 and 1,536 evidence-bearing questions.
  Category 5 (446 adversarial questions with evidence) is reported separately.
- Metrics use the same strict Recall-All and corrected NDCG implementation as
  the LongMemEval runner for consistency. This is not LoCoMo's official F1 QA
  score and should not be compared with published LoCoMo QA leaderboards.

### Core categories (1–4)

| Retriever | Recall-Any@5 | Recall-All@5 | NDCG-Any@5 | Recall-Any@10 | Recall-All@10 | NDCG-Any@10 | q/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Memoria windowed** | **90.2%** | **78.3%** | **0.7819** | **96.7%** | **87.5%** | **0.8120** | 5.4 |
| Flat BM25 | 74.3% | 62.2% | 0.6183 | 86.6% | 73.3% | 0.6610 | 326.4 |
| Memoria legacy | 62.4% | 50.9% | 0.4826 | 80.3% | 68.4% | 0.5459 | 15.3 |
| MiniLM dense | 52.7% | 41.5% | 0.4081 | 68.3% | 55.7% | 0.4612 | 81.0 |

### Category diagnosis

| Category | n | Windowed R-All@5 | Legacy R-All@5 | BM25 R-All@5 | Gain vs legacy |
|---|---:|---:|---:|---:|---:|
| Single-hop | 282 | 34.4% | 21.6% | 15.6% | +12.8 pp |
| Temporal | 321 | 84.7% | 67.3% | 60.4% | +17.4 pp |
| Multi-hop | 92 | 50.0% | 27.2% | 34.8% | +22.8 pp |
| Open-domain | 841 | 93.7% | 57.1% | 81.6% | +36.6 pp |
| Adversarial diagnostic | 446 | 93.0% | 57.8% | 82.5% | +35.2 pp |

The diagnosis was correct: whole-session dense embeddings and a cross-encoder
limited to the session prefix were suppressing evidence later in a session.
Windowing makes the neural signals meaningful, while parent-level RRF prevents
them from erasing exact lexical hits. The remaining weakness is strict
single-hop and multi-hop Recall-All, where questions often have several labeled
evidence sessions and every one must fit inside the top k.

### Frozen generalization-hardening audit

A second change was developed without reading LoCoMo questions, evidence labels,
or per-category examples during implementation. It adds three general rules:

1. only explicit multi-entity comparisons receive entity-scoped query probes;
2. a parent can cover an entity facet only when that entity occurs in one of its
   windows; and
3. equal lexical, dense, or reranker scores receive equal ranks instead of being
   ordered by corpus position.

The implementation was frozen at retriever SHA-256
`e0b026def0f7afb5c7245532cd3afcd387fe9316efa230d19a91bde0a8fbe12e` after
synthetic tests passed. The complete clean-install suite then passed 57/57, and
the full LongMemEval-S run exactly preserved 94.3% Recall-All@5 and 97.1%
Recall-All@10. Only after those checks was LoCoMo run once.

The frozen LoCoMo result was neutral: core Recall-All@5 remained 78.3%,
Recall-All@10 remained 87.5%, and NDCG-Any@10 moved from 0.811982 to 0.811984.
Single-hop gained one Recall-All@5 pass, while open-domain lost one, leaving the
aggregate unchanged. Recall-Any@5 lost one core pass. This is useful evidence
that the generic change did not regress strict aggregate recall, but it is not
evidence of a meaningful LoCoMo improvement. The raw post-freeze artifact is
`results/2026-08-11/locomo_memoria_faceted_freeze.json`.

This audit reduces direct example-fitting risk but does not make LoCoMo blind:
the benchmark's earlier aggregate and category results were already known. A
new external or privately held conversation set is still required for an
unbiased generalization claim.

## PerLTQA untouched external holdout

### Protocol

- Dataset: the official updated English v2 files at upstream commit
  `8d9e19868e239740ef701e603ec205cd581f221b`, released after the repository's
  December 2025 consistency update.
- Scope: all 8,593 questions across 32 QA character groups. The corresponding
  memory file contains 141 character banks; each question searches only its
  character's bank, matching the paper's task definition.
- Scored unit: one native profile attribute, relationship, event, or dialogue
  memory record. Long events are split into overlapping four-sentence windows;
  dialogues use four-turn windows. Both use stride two and collapse to the
  native record ID before scoring.
- Metrics: the paper's Recall@1, Recall@2, Recall@3, and Recall@5. Every row has
  exactly one reference-memory label.
- Data integrity: 102 labels cannot be retrieved because their IDs are absent
  from the supplied v2 memory banks—95 relationship labels whose character has
  only an unindexed relationship narrative, and seven absent event IDs. The
  strict scope counts these as misses; valid-gold metrics exclude only these
  impossible rows.
- Controls: windowed BM25 and MiniLM receive the identical chunks and
  parent-record collapse. No paid API or answer generator is used.

The adapter was frozen at SHA-256
`7028cf87c740dd99706d7b2c0371fdd1f88ed4bf9d622feaee5aad2112c73716`
before retrieval scoring. The initial run exposed a production shape mismatch
for a question naming more entities than the configured facet cap. That run
terminated without writing a result. The invariant was fixed, a regression test
was added, the full 64-test suite passed, and the evaluation restarted from row
one with retriever SHA-256
`433a7bd7f1bef901486b6eecd14d4d1b637f44ec64e0c2c53b389d9ca4831104`.
No score-driven change followed.

### Results

| Retriever | Strict R@1 | Strict R@2 | Strict R@3 | Strict R@5 | Valid-gold R@5 | q/s |
|---|---:|---:|---:|---:|---:|---:|
| **Memoria balanced** | **50.7%** | **71.4%** | **77.5%** | **84.4%** | **85.4%** | 9.0 |
| Windowed BM25 | 48.8% | 67.7% | 73.6% | 80.5% | 81.5% | 140.7 |
| Windowed MiniLM | 42.1% | 60.6% | 66.1% | 71.9% | 72.7% | 63.1 |

| Native memory type | Rows | Missing gold IDs | Strict R@5 | Valid-gold R@5 |
|---|---:|---:|---:|---:|
| Profile | 357 | 0 | 77.9% | 77.9% |
| Social relationship | 897 | 95 | 78.4% | 87.7% |
| Event | 4,501 | 7 | 91.8% | 92.0% |
| Dialogue | 2,838 | 0 | 75.3% | 75.3% |

This is the first genuinely untouched external result in this audit. It supports
the narrower claim that the frozen hybrid/windowed retrieval design generalizes
beyond LongMemEval and LoCoMo and beats local lexical-only and dense-only
controls. It does not demonstrate that comparison faceting alone caused the
gain, and the relatively weak dialogue score identifies the next research
problem without authorizing tuning on PerLTQA.

The paper reports BM25 R@5 of 89.5% and trained DPR R@5 of 91.9% on its original
1,719-row test split. Those values are context rather than direct controls: the
repository does not publish that split, the v2 data were subsequently corrected,
and this audit evaluates the complete updated English set with a documented
windowed rendering. The local same-data controls above are the valid comparison.

## Validity limits

1. The benchmark-facing API receives already segmented raw conversations. It
   does not test extraction, graph construction, consolidation, or graph
   traversal. The grouped retrieval method is general production code, but this
   benchmark exercises it directly rather than through `Memoria.remember()`.
2. No reader model generated answers and no official LLM judge ran. Therefore
   these results cannot be compared with LongMemEval QA accuracy, LoCoMo answer
   F1/LLM-judge scores, MemoryAgentBench accuracy, or BEAM answer scores.
3. LongMemEval and LoCoMo are public evaluation sets. Historical retrieval choices were
   developed with LongMemEval, and the new grouped design was developed after
   observing LoCoMo's legacy category breakdown. Both numbers are engineering
   results, not blind held-out estimates. PerLTQA was held untouched until the
   retriever and adapter protocol were frozen, so it supplies an external
   generalization check, although it remains synthetic public data rather than
   a private real-user holdout.
4. Throughput is end-to-end query time after model availability on one Apple M3
   Max process. Windowed LoCoMo retrieval reranks 30 chunks per question and is
   slower than the legacy path. BM25 does not load neural models; model download
   time is not included in the per-query timer.
5. The run used an uncommitted audited working tree based on commit
   `3fb6eacd24034199ee463344a07f67f2e05f08ee`. Exact environment and dataset
   hashes are in the result manifest.

## Reproduction

```bash
pip install -e ".[dev]"

python benchmarks/longmemeval_final.py \
  --data benchmarks/data/longmemeval_s_cleaned.json \
  --scope official --mode balanced
python benchmarks/longmemeval_final.py \
  --data benchmarks/data/longmemeval_s_cleaned.json \
  --scope official --retriever flat-bm25
python benchmarks/longmemeval_final.py \
  --data benchmarks/data/longmemeval_s_cleaned.json \
  --scope official --retriever minilm

python benchmarks/locomo_bench.py \
  --data benchmarks/data/locomo10.json \
  --mode balanced --retriever memoria --profile windowed
python benchmarks/locomo_bench.py \
  --data benchmarks/data/locomo10.json \
  --mode balanced --retriever memoria --profile legacy
python benchmarks/locomo_bench.py \
  --data benchmarks/data/locomo10.json \
  --retriever flat-bm25
python benchmarks/locomo_bench.py \
  --data benchmarks/data/locomo10.json \
  --retriever minilm

python benchmarks/download_perltqa.py
python benchmarks/perltqa_bench.py \
  --retriever memoria --mode balanced \
  --output benchmarks/results/2026-08-11/perltqa_en_v2_memoria_frozen.json
python benchmarks/perltqa_bench.py \
  --retriever window-bm25 \
  --output benchmarks/results/2026-08-11/perltqa_en_v2_window_bm25.json
python benchmarks/perltqa_bench.py \
  --retriever window-minilm \
  --output benchmarks/results/2026-08-11/perltqa_en_v2_window_minilm.json
```

Raw outputs and the machine-readable manifest are in
[`benchmarks/results/2026-08-11`](results/2026-08-11/).
