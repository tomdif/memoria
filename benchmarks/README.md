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

Both support `--retriever memoria`, `--retriever flat-bm25`, and
`--retriever minilm` for same-data controls. See
`BENCHMARK_REPORT_2026-08-11.md` for the latest protocol, results, limitations,
and exact commands.

PerLTQA uses the analogous controls named `window-bm25` and `window-minilm` so
all three systems receive the same four-item windows and parent-record collapse.

`longmemeval_bench.py`, `longmemeval_optimized.py`, and `headtohead_bench.py`
are historical experiment runners. The result JSON files at the root of this
directory were produced by those older protocols. They include scopes or metric
implementations that do not match the audited official-compatible runner and
must not be used as current claims.
