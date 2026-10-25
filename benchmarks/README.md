# Benchmarks

The latest full run is [October 25, 2026](BENCHMARK_RERUN_2026-10-25.md):
10 retrieval configurations, seven matched-control/ablation arms, and 2,000
fresh reader/judge evaluations. Every retrieval metric, subgroup score, and
scoring count matches its original artifact. QA scores vary despite identical
retrieval lists; see the report for the comparison and judge-limit caveat.

## Runners and scope

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

The LongMemEval and LoCoMo runners support `--retriever memoria`,
`--retriever flat-bm25`, and `--retriever minilm` for same-data controls. See
[the October rerun](BENCHMARK_RERUN_2026-10-25.md) for current results and
[the August report](BENCHMARK_REPORT_2026-08-11.md) for the original protocol.

PerLTQA uses the analogous controls named `window-bm25` and `window-minilm` so
all three systems receive the same four-item windows and parent-record collapse.

`controls_ce.py` and `controls_ablation.py` accept `--output` for independently
stored reruns. `longmemeval_qa.py` accepts `--output-dir` for retrieval maps,
resumable QA records, and reports. Choose a new directory for a fresh evaluation;
existing QA records are resumed. Plain and CoT variants have separate files.

From the repository root, `python benchmarks/compare_rerun.py benchmarks/results/2026-10-25`
compares all published retrieval metrics,
denominators, subgroup scores, controls, and QA labels with the original run.
It reports missing files and incomplete QA arms explicitly. Dated result
directories preserve the original artifacts.

These runners exercise raw-session/window retrieval. They do not evaluate
production extraction, graph ingestion, or consolidation. The separate
`product_reliability_probe.py` measures synthetic database overhead.

## Prepare a fresh run

Run all commands below from the repository root in one POSIX shell, with the
Memoria virtual environment active. The canonical runners use core dependencies;
the `llm` extra adds the Anthropic client for QA. The `bench` extra supplies
ChromaDB for historical experiments and is not needed for this suite.

```bash
python -m pip install -e '.[llm]'
python benchmarks/download_data.py
python -c 'from benchmarks.locomo_bench import download; download()'
python benchmarks/download_perltqa.py

mkdir -p benchmarks/results
export MEMORIA_BENCH_RUN="$(mktemp -d "$PWD/benchmarks/results/rerun-XXXXXXXX")"
```

Check dataset bytes against the audited manifest before comparing results:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path('benchmarks/results/2026-10-25/manifest.json').read_text())
for name, expected in manifest['datasets'].items():
    data = (Path('benchmarks/data') / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
    print(f'{name}: verified')
PY
```

The [manifest](results/2026-10-25/manifest.json) also records Python/package
versions and both Hugging Face model revisions. Match those for a strict
replication; an unpinned installation may resolve different versions. Models
download on first use. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` only
after the required revisions are cached.

## Run all retrieval configurations and controls

Absolute output paths avoid the legacy LoCoMo runner's script-relative path
behavior and keep every result in the same new directory.

```bash
python benchmarks/longmemeval_final.py --scope official --mode balanced \
  --output "$MEMORIA_BENCH_RUN/longmemeval_s_balanced.json"
python benchmarks/longmemeval_final.py --scope official --retriever flat-bm25 \
  --output "$MEMORIA_BENCH_RUN/longmemeval_s_flat_bm25.json"
python benchmarks/longmemeval_final.py --scope official --retriever minilm \
  --output "$MEMORIA_BENCH_RUN/longmemeval_s_minilm.json"

python benchmarks/locomo_bench.py --mode balanced --profile windowed \
  --output "$MEMORIA_BENCH_RUN/locomo_memoria_windowed.json"
python benchmarks/locomo_bench.py --mode balanced --profile legacy \
  --output "$MEMORIA_BENCH_RUN/locomo_memoria_legacy.json"
python benchmarks/locomo_bench.py --retriever flat-bm25 \
  --output "$MEMORIA_BENCH_RUN/locomo_flat_bm25.json"
python benchmarks/locomo_bench.py --retriever minilm \
  --output "$MEMORIA_BENCH_RUN/locomo_minilm.json"

python benchmarks/perltqa_bench.py --retriever memoria \
  --output "$MEMORIA_BENCH_RUN/perltqa_en_v2_memoria.json"
python benchmarks/perltqa_bench.py --retriever window-bm25 \
  --output "$MEMORIA_BENCH_RUN/perltqa_en_v2_window_bm25.json"
python benchmarks/perltqa_bench.py --retriever window-minilm \
  --output "$MEMORIA_BENCH_RUN/perltqa_en_v2_window_minilm.json"

python benchmarks/controls_ce.py --output "$MEMORIA_BENCH_RUN/controls_ce.json"
python benchmarks/controls_ablation.py --output "$MEMORIA_BENCH_RUN/controls_ablation.json"
```

## Run fresh QA and compare

These commands make paid API calls using `ANTHROPIC_API_KEY`. The October run
cost an estimated $34.31 for all four 500-question arms; actual charges depend
on provider pricing and token usage. Original prompts, model IDs, and token
budgets are defined in `longmemeval_qa.py`.

```bash
python benchmarks/longmemeval_qa.py --stage retrieve --output-dir "$MEMORIA_BENCH_RUN"
python benchmarks/longmemeval_qa.py --stage qa --arm memoria --output-dir "$MEMORIA_BENCH_RUN"
python benchmarks/longmemeval_qa.py --stage qa --arm hybrid-ce-dual --output-dir "$MEMORIA_BENCH_RUN"
python benchmarks/longmemeval_qa.py --stage qa --arm memoria --cot --output-dir "$MEMORIA_BENCH_RUN"
python benchmarks/longmemeval_qa.py --stage qa --arm hybrid-ce-dual --cot --output-dir "$MEMORIA_BENCH_RUN"
python benchmarks/longmemeval_qa.py --stage report --output-dir "$MEMORIA_BENCH_RUN" \
  > "$MEMORIA_BENCH_RUN/qa_report.txt"
python benchmarks/compare_rerun.py "$MEMORIA_BENCH_RUN" \
  --output "$MEMORIA_BENCH_RUN/comparison.json"
```

QA appends records and resumes completed question IDs. Keep the same directory
to resume an interrupted run; create a new one for an independent run. Changing
models or prompts while reusing a directory mixes protocols. For a complete
comparison, `missing` must be empty, all four QA `complete` fields must be true,
and both retrieval maps must contain the expected 500 question IDs. A successful
comparison command alone does not establish completeness.

## Product overhead and historical artifacts

```bash
python benchmarks/product_reliability_probe.py --source-tree .
```

This probe uses temporary databases and fixed extraction/reranking. It measures
commits, document-loading SQL queries, and ingestion time; it does not measure
model inference or memory accuracy. The [engineering review](../docs/ENGINEERING_REVIEW_2026-10-25.md)
records the before/after comparison.

`longmemeval_bench.py`, `longmemeval_optimized.py`, and `headtohead_bench.py`
are historical experiment runners. Their root-level results include scopes or
metric implementations that differ from the audited runner and must not be used
as current claims. Root-level `results_controls_*`, `qa_retrieval_*`, and
`results_qa_*` files are the published August controls and QA baseline.
