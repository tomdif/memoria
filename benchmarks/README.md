# Benchmarks

The canonical audited runners are:

- `longmemeval_final.py`: LongMemEval-S session retrieval, including the
  official retrieval scope and metric definitions.
- `locomo_bench.py`: LoCoMo session-evidence retrieval diagnostic, explicitly
  separated from LoCoMo's generated-answer QA score. Its default `windowed`
  profile uses overlapping turn windows, date/image-caption enrichment,
  lexical+dense reciprocal-rank fusion, comparison-query facets, chunk
  reranking, and parent-session evidence coverage. Pass `--profile legacy` to
  reproduce the old whole-session path.
- `perltqa_bench.py`: PerLTQA English v2 native-memory retrieval. It windows
  long event/dialogue records, collapses them to official memory IDs, and
  reports the paper's Recall@1/2/3/5 over both strict and valid-gold scopes.
  `download_perltqa.py` pins the official upstream commit and verifies both
  dataset checksums.

The LongMemEval and LoCoMo runners support `--retriever memoria`, `--retriever flat-bm25`, and
`--retriever minilm` for same-data controls. See
[the October rerun](BENCHMARK_RERUN_2026-10-25.md) for current results and
[the August report](BENCHMARK_REPORT_2026-08-11.md) for the original protocol.

PerLTQA uses the analogous controls named `window-bm25` and `window-minilm` so
all three systems receive the same four-item windows and parent-record collapse.

`controls_ce.py` and `controls_ablation.py` accept `--output` for independently
stored reruns. `longmemeval_qa.py` accepts `--output-dir` for retrieval maps,
resumable QA records, and reports. Choose a new directory for a fresh evaluation;
existing QA records are resumed. Plain and CoT variants have separate files.

`compare_rerun.py results/2026-10-25` compares all published retrieval metrics,
denominators, subgroup scores, controls, and QA labels with the original run.
It reports missing files and incomplete QA arms explicitly. Dated result
directories preserve the original artifacts.

`longmemeval_bench.py`, `longmemeval_optimized.py`, and `headtohead_bench.py`
are historical experiment runners. Their root-level results include scopes or
metric implementations that differ from the audited runner and must not be used
as current claims. Root-level `results_controls_*`, `qa_retrieval_*`, and
`results_qa_*` files are the published August controls and QA baseline.
