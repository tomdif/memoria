# Memoria benchmark rerun — 2026-10-25

A full rerun of the published retrieval benchmarks, matched controls, component
ablations, and plain/chain-of-thought LongMemEval-S QA on the reliability branch.
Product code was frozen at `39a6879e0eef74b9ffd06b1b9186d73aa6d9acf7`.
Only benchmark output routing and response metadata were added for this rerun;
no retrieval parameters were tuned against the results.

## Retrieval

| Dataset / system | Scored rows | Previous Recall@5 | Rerun Recall@5 | Rerun Recall@10 |
|---|---:|---:|---:|---:|
| LongMemEval-S, Memoria balanced | 419 | 94.3% | 94.3% | 97.1% |
| LoCoMo, Memoria windowed (categories 1–4) | 1,536 | 78.3% | 78.3% | 87.5% |
| PerLTQA English v2, Memoria (strict) | 8,593 | 84.4% | 84.4% | — |

10 of 10 retrieval configurations reproduce every reported metric and subgroup score exactly.
Together with the seven control/ablation arms, the audit compares 747 metric
and scoring-count fields with no differences.
LongMemEval and LoCoMo use strict Recall-All@K; PerLTQA has one reference
memory per question and uses Recall@K. LoCoMo includes all 1,982 evidence-labeled
questions in the artifact, with 1,536 non-adversarial questions in the headline
table. PerLTQA includes all 8,593 questions; its 102 absent gold IDs count as
strict misses. Valid-gold scores are also retained in the raw artifact.

### Lexical, dense, and legacy controls

| System | Previous Recall@5 | Rerun Recall@5 |
|---|---:|---:|
| longmemeval_s_flat_bm25 | 74.2% | 74.2% |
| longmemeval_s_minilm | 85.4% | 85.4% |
| locomo_flat_bm25 | 62.2% | 62.2% |
| locomo_minilm | 41.5% | 41.5% |
| locomo_memoria_legacy | 50.9% | 50.9% |
| perltqa_en_v2_window_bm25 | 80.5% | 80.5% |
| perltqa_en_v2_window_minilm | 71.9% | 71.9% |

### Matched cross-encoder controls and ablations

All arms below use the same 419-question LongMemEval official retrieval scope,
the same models, and the same candidate/reranking budgets as the original run.
All seven arms reproduce their scores and question-type breakdowns exactly.
See the [original controls addendum](CONTROLS_ADDENDUM.md) for arm definitions.

| Arm | Previous Recall-All@5 | Rerun Recall-All@5 | Rerun Recall-All@10 |
|---|---:|---:|---:|
| bm25-ce | 82.6% | 82.6% | 85.4% |
| minilm-ce | 87.6% | 87.6% | 95.2% |
| hybrid-ce | 89.5% | 89.5% | 97.1% |
| hybrid-ce-dual | 90.5% | 90.5% | 95.9% |
| blend-only | 93.6% | 93.6% | 96.7% |
| exp-only | 90.2% | 90.2% | 96.2% |
| exp+blend | 94.3% | 94.3% | 97.1% |

## End-to-end QA

Each of the four arms ran 500 fresh reader/judge evaluations, including the 30
abstention questions. No prior answer or judge record was reused. The reader
remained `claude-haiku-4-5-20251001`; the judge remained `claude-sonnet-5`.
Original prompts, reader temperature 0, token budgets, disabled judge thinking,
and official judge parsing were retained. Judge temperature remains the
provider default because this model rejects an explicit temperature.

| Arm | Previous accuracy | Rerun accuracy | Correct / 500 | Change | Abstention accuracy |
|---|---:|---:|---:|---:|---:|
| Memoria — plain | 80.8% | 80.2% | 401 | -0.6 pp | 96.7% |
| Matched control — plain | 75.6% | 76.4% | 382 | +0.8 pp | 90.0% |
| Memoria — CoT | 86.0% | 86.6% | 433 | +0.6 pp | 90.0% |
| Matched control — CoT | 82.8% | 83.8% | 419 | +1.0 pp | 90.0% |

Both arms reproduce all 500 ordered top-five retrieval lists exactly. Thus
changes in QA labels occur after evidence selection, in repeated generation
and judging. This rerun does not establish a QA accuracy improvement from the
reliability changes. A single paired replication cannot separate provider drift
from stochastic reader/judge behavior.

| Arm | Previously correct, now wrong | Previously wrong, now correct | Reader truncations | Judge truncations |
|---|---:|---:|---:|---:|
| qa_memoria | 15 | 12 | 0 | 0 |
| qa_cot_memoria | 12 | 15 | 0 | 0 |
| qa_hybrid-ce-dual | 18 | 22 | 0 | 0 |
| qa_cot_hybrid-ce-dual | 15 | 20 | 0 | 1 |

No reader answer was truncated. One CoT matched-control judgment
(question `6456829e`) reached the original 10-token limit without producing
a yes/no answer. The unchanged official parser records it as incorrect.
It was not selectively rejudged; resolving it could raise that control score
by at most 0.2 percentage points (83.8% to 84.0%).

279 answer texts were exactly repeated from August; 0 of those
received a different judge label. This checks exact text matches only; it does
not isolate judging variability for answers whose wording changed.

Estimated QA API cost: **$34.31** across all four full arms, based on
29,686,574/478,512 reader input/output tokens and
1,074,890/7,922 judge input/output tokens.
This is a usage-based estimate, not an invoice. The two tiny model-availability
probes are excluded. Rates were verified against the official
[Haiku 4.5](https://platform.claude.com/docs/en/models/haiku-4-5/overview) and
[Sonnet 5](https://platform.claude.com/docs/en/models/sonnet-5/overview) pages.

## Reproducibility and limits

- All four dataset SHA-256 values match the original artifacts. Model revisions,
  Python/package versions, source hashes, and hardware are recorded in the manifest.
- The machine is an Apple M3 Max with 36 GiB RAM. Local retrieval uses cached
  models with Hugging Face network access disabled.
- Throughput is recorded but is not a controlled speed comparison: lexical
  baselines and remote QA orchestration shared the machine with neural retrieval.
- These published benchmarks exercise raw-session/window retrieval, not the
  production extraction, graph-ingestion, or consolidation path. Their previous
  post-hoc development and dataset-label limitations still apply.
- All 194 product tests pass. The separate synthetic product probe was rerun;
  it still verifies one ingestion commit and three recall document-loading SELECTs
  for 1,000 facts. It is not a memory-accuracy benchmark.

## Artifacts and reproduction

- [Manifest](results/2026-10-25/manifest.json)
- [Complete numeric comparison](results/2026-10-25/comparison.json)
- [QA report, including per-ability results](results/2026-10-25/qa_report.txt)
- [QA record integrity and stop-reason audit](results/2026-10-25/qa_integrity_audit.json)
- [Neural retrieval command log](results/2026-10-25/neural_jobs.json)
- [Lexical command log](results/2026-10-25/lexical_jobs.json)
- [QA command log](results/2026-10-25/qa_jobs.json)
- [Synthetic product probe](results/2026-10-25/product_reliability_probe.json)

The [benchmark guide](README.md) provides the complete portable command
sequence: dataset downloads and checksum verification, all 17 retrieval/control
configurations, all four QA arms, and the comparison. It creates a new output
directory so historical results remain intact. QA resumes existing question
records; a new directory is required for an independent evaluation.

To inspect the committed comparison without rerunning models or calling APIs:

```bash
python benchmarks/compare_rerun.py benchmarks/results/2026-10-25
```

Queue commands, including lexical/dense and legacy controls, are captured in
the job artifacts with execution-machine paths. The original August result
artifacts remain unchanged. Product tests and package checks passed in the
[CI run for the benchmark commit](https://github.com/tomdif/memoria/actions/runs/40213835893).
