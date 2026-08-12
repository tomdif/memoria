# Addendum — cross-encoder attribution controls (2026-08-12)

The 2026-08-11 report compared Memoria's balanced session retriever against
flat BM25 and plain MiniLM dense retrieval. Neither baseline used a reranker,
so the headline gap could not be attributed: it might have been nothing more
than "add an off-the-shelf cross-encoder." These controls close that gap.

## Protocol

Identical to the audited LongMemEval-S runner in every respect: same official
retrieval scope (419 rows), same dataset file (SHA-256
`d6f21ea9…3a442`), same session documents from
`build_session_documents`, same official metric implementation
(`retrieval_metrics.py`). Every control uses the **same cross-encoder**
(`cross-encoder/ms-marco-MiniLM-L-12-v2`) at the **same candidate depth as
Memoria balanced** (`_RERANK_CONFIG[BALANCED]` → K=15), with candidates ordered
purely by cross-encoder score (the standard retrieve-then-rerank pipeline).

Ladder of controls, each adding one Memoria design element:

1. `bm25-ce` — official whitespace BM25 top-15 → CE.
2. `minilm-ce` — MiniLM dense top-15 → CE.
3. `hybrid-ce` — Memoria's stage-1 fusion (0.70·dense + 0.30·BM25, Memoria
   tokenizer, dense pool 60, **no** query expansion) top-15 → CE.
4. `hybrid-ce-dual` — same candidates; CE score = max over the user-only[:512]
   and user+assistant[:512] renderings (Memoria's dual-pass), still pure CE
   ordering.

Two ablations then isolate the remaining pipeline elements:

- `blend-only` — hybrid candidates + dual-pass CE + Memoria's final fusion
  (0.40·CE-minmax + 0.60·stage1-norm) but **no** expansion.
- `exp-only` — stage-1b conditional single-probe query expansion + dual-pass
  CE, pure CE ordering (no blend).
- `exp+blend` — both. This arm reproduced the audited Memoria balanced result
  digit-for-digit (94.3 / 98.1 / 97.1, NDCG 0.9414 / 0.9478), validating the
  reimplementation.

Runners: `benchmarks/controls_ce.py`, `benchmarks/controls_ablation.py`.
Raw outputs: `benchmarks/results_controls_ce.json`,
`benchmarks/results_controls_ablation.json`.

## Results (official scope, 419 rows)

| System | R-Any@5 | R-All@5 | NDCG@5 | R-Any@10 | R-All@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| bm25-ce | 94.3% | 82.6% | 0.8639 | 95.0% | 85.4% | 0.8692 |
| minilm-ce | 96.4% | 87.6% | 0.9001 | 98.8% | 95.2% | 0.9163 |
| hybrid-ce | 96.7% | 89.5% | 0.9070 | 99.0% | 97.1% | 0.9231 |
| hybrid-ce-dual | 96.9% | 90.5% | 0.9137 | 98.6% | 95.9% | 0.9250 |
| blend-only | 98.1% | 93.6% | 0.9380 | 99.3% | 96.7% | 0.9457 |
| exp-only | 96.9% | 90.2% | 0.9130 | 98.6% | 96.2% | 0.9251 |
| **Memoria balanced** (= exp+blend) | **98.1%** | **94.3%** | **0.9414** | 99.0% | **97.1%** | **0.9478** |

Attribution of the +6.7 pp Recall-All@5 over the strongest commodity pipeline
(minilm-ce, 87.6%):

| Design element | Δ R-All@5 |
|---|---:|
| Hybrid dense+BM25 candidate fusion (vs dense-only) | +1.9 pp |
| Dual-pass CE rendering (user-only ∨ full-text) | +1.0 pp |
| Final score fusion 0.40·CE + 0.60·stage1 (vs pure CE ordering) | +3.1 pp |
| Conditional query expansion (only in combination with the fusion) | +0.7 pp |

Query expansion **alone** is neutral-to-negative under pure CE ordering
(90.2% vs 90.5% without it); its contribution only materializes through the
stage-1 term of the blended final score.

## Per-ability, Memoria vs strongest pure-CE control (hybrid-ce-dual), R-All@5

| Ability | n | Memoria | hybrid-ce-dual | Δ pp | ≈rows |
|---|---:|---:|---:|---:|---:|
| Knowledge update | 72 | 100.0% | 95.8% | +4.2 | 3 |
| Multi-session | 121 | 94.2% | 90.1% | +4.1 | 5 |
| Single-session assistant | 5 | 100.0% | 100.0% | 0.0 | 0 |
| Single-session preference | 30 | 96.7% | 93.3% | +3.4 | 1 |
| Single-session user | 64 | 100.0% | 95.3% | +4.7 | 3 |
| Temporal reasoning | 127 | 87.4% | 84.3% | +3.1 | 4 |

The edge is broad — roughly 16 additional strict passes spread across five of
six abilities — rather than concentrated in one question type.

## Conclusion

Memoria's session-retrieval design contributes measurably beyond a commodity
retrieve-then-rerank pipeline built from the identical models: **+3.8 pp strict
Recall-All@5 over the strongest cross-encoder control (94.3% vs 90.5%) and
+6.7 pp over dense→CE**. The single largest factor is not the candidate
generation but the decision **not to let the cross-encoder fully override
first-stage evidence** (0.40/0.60 score fusion, +3.1 pp); hybrid candidates,
dual-pass rendering, and conditional expansion contribute the remainder.

## Caveats

1. Single dataset and scope: LongMemEval-S official retrieval scope only. The
   fusion weights (0.70/0.30, 0.40/0.60) were historically developed with
   LongMemEval in view; this decomposition attributes the gap but is not a
   blind held-out estimate. The LoCoMo/PerLTQA runs in the main report address
   generalization of the overall retriever, not of this decomposition.
2. Re-cleaned data: the September-2025 cleaned LongMemEval-S file; numbers are
   not comparable to results on the original release.
3. K=15 CE depth is Memoria's own balanced setting. Deeper commodity pipelines
   (e.g. CE over top-50) were not run; the controls answer "same budget, same
   models, standard architecture," not "best possible commodity pipeline."
4. Retrieval metrics only; no answer generation. Nothing here is comparable to
   LongMemEval QA accuracy leaderboards.
5. Three of 419 rows have six evidence sessions and cannot pass Recall-All@5;
   differences of ~1 pp correspond to ~4 rows.
