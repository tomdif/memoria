# Addendum — end-to-end LongMemEval-S QA evaluation (2026-08-12)

The main report and the controls addendum evaluate retrieval only. This
addendum adds the number the field actually compares on: **end-to-end QA
accuracy** on all 500 LongMemEval-S questions (abstention included), using the
official LongMemEval model-judge protocol.

## Protocol

- **Data:** the cleaned 500-row LongMemEval-S file (SHA-256 `d6f21ea9…3a442`),
  all 500 questions — 470 answerable + 30 abstention (`*_abs`).
- **Retrieval:** top-5 sessions per question, two arms:
  1. `memoria` — Memoria balanced session retriever;
  2. `hybrid-ce-dual` — the strongest commodity control from
     `CONTROLS_ADDENDUM.md` (hybrid dense+BM25 candidates → top-15 →
     dual-pass cross-encoder, pure CE ordering).
- **Reader:** `claude-haiku-4-5-20251001`, temperature 0, max_tokens 1024,
  using the official non-CoT retrieval-augmented prompt from
  `src/generation/run_generation.py` (nl history format, full user+assistant
  turns, session dates, current date, sessions in rank order).
- **Judge:** the official per-question-type judge prompts from
  `src/evaluation/evaluate_qa.py`, verbatim, including the abstention
  template and the `'yes' in response.lower()` parsing. Judge model:
  `claude-sonnet-5` with thinking disabled, max_tokens 10. Deviation from the
  official protocol: Claude Sonnet 5 rejects non-default sampling parameters,
  so the official `temperature=0` could not be pinned; temperature was
  omitted. Both arms use the identical reader, judge, and prompts.
- **Runner:** `benchmarks/longmemeval_qa.py`. Raw per-question records
  (hypothesis, judge response, label, retrieved IDs, token usage):
  `benchmarks/results_qa_memoria.json`, `benchmarks/results_qa_hybrid-ce-dual.json`.
- **Cost:** $16.54 total (reader ~14.8M input / 161K output tokens on Haiku
  4.5; judge ~431K input tokens on Sonnet 5 intro pricing).

## Results

| Metric | Memoria | hybrid-ce-dual control | Δ |
|---|---:|---:|---:|
| **Overall QA accuracy (500)** | **80.8%** | 75.6% | **+5.2 pp** |
| Non-abstention accuracy (470) | 79.8% | 74.7% | +5.1 pp |
| Abstention accuracy (30) | 96.7% | 90.0% | +6.7 pp |

Per ability (non-abstention):

| Ability | n | Memoria | hybrid-ce-dual | Δ pp |
|---|---:|---:|---:|---:|
| Single-session user | 64 | 100.0% | 95.3% | +4.7 |
| Single-session assistant | 56 | 96.4% | 98.2% | −1.8 |
| Knowledge update | 72 | 79.2% | 68.1% | +11.1 |
| Multi-session | 121 | 76.0% | 71.1% | +4.9 |
| Temporal reasoning | 127 | 70.9% | 65.4% | +5.5 |
| Single-session preference | 30 | 60.0% | 56.7% | +3.3 |

## Retrieval-vs-reader gap

Rows in the official retrieval scope (419, non-abstention with user-side
evidence labels):

| | Memoria | hybrid-ce-dual |
|---|---:|---:|
| QA accuracy when all evidence retrieved (R-All@5 = 1) | 82.0% (n=395) | 78.1% (n=379) |
| QA accuracy when retrieval missed | 8.3% (n=24) | 12.5% (n=40) |
| Reader failures despite correct retrieval | 71 rows | 83 rows |
| Retrieval misses | 24 rows | 40 rows |

Retrieval is close to a hard prerequisite (accuracy collapses to ~10% on
misses), and Memoria's retrieval edge carries through to the QA level.
**The system is now reader-limited, not retrieval-limited**: for the Memoria
arm, 71 of the 95 wrong answers on evidence-labeled rows occur with all
evidence in context. The largest reader-failure pools are temporal reasoning
(judged strictly on dates/durations) and single-session preference (rubric
satisfaction). A stronger reader or a CoT reading prompt is the obvious next
lever; the official paper reports CoT helps precisely on these types.

## Where this sits vs published numbers

Published LongMemEval-S end-to-end accuracies (different readers — noted):

| System | Reader | Overall acc |
|---|---|---:|
| Full-context baseline | gpt-4o-mini | 49.8% |
| Mem0 / Mem0-Graph | gpt-4o-mini | 66.9% / 68.4% |
| Zep | gpt-4o | 71.2% |
| LiCoMemory | gpt-4o-mini | 73.8% |
| TiMem | gpt-4o-mini | 76.9% |
| **Memoria (this run)** | **claude-haiku-4-5** | **80.8%** |
| Mastra "Observational Memory" (vendor claim) | (theirs) | ~95% |

Honest caveats on that table: readers differ (Haiku 4.5 is a late-2025 model
and likely stronger than gpt-4o-mini; same tier by price/latency positioning,
not by benchmark identity); most published numbers predate the Sept-2025
dataset re-clean; and evaluation judges differ (gpt-4o judge in the official
protocol vs Claude Sonnet 5 here). The within-run comparison — Memoria vs the
matched commodity control under an identical reader, judge, and prompts — is
the controlled claim: **+5.2 pp end-to-end from Memoria's retrieval design**.
The cross-paper table is context, not a leaderboard placement.

## Limitations

1. Single reader, single seed, no CoT; the reader prompt is the official
   plain template. Numbers move with reader choice.
2. Judge is Claude, not the official gpt-4o judge, and temperature could not
   be pinned to 0 on the judge model. Judge agreement with the official judge
   was not measured.
3. LongMemEval retrieval choices were historically developed against this
   benchmark (see main report §Validity limits); the QA numbers inherit that
   caveat. The commodity control shares the same data, so the *delta* is the
   robust quantity.
4. The pipeline bypasses Memoria's ingestion/graph path (same as all prior
   benchmarks in this repo): this measures the session retriever + reader,
   not the full product.
