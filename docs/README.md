# Memoria documentation

Memoria 0.3 provides local, scoped memory through Python, CLI, stdio MCP, HTTP,
and lifecycle hooks. Start with the [project README](../README.md) for installation
and editor setup.

| Guide | Purpose |
|---|---|
| [API reference](API.md) | Python calls, structured recall, HTTP endpoints, MCP, and scope behavior |
| [Storage operations](STORAGE.md) | Quotas, complete backups, integrity checks, retention, restore, and cleanup |
| [Upgrading to 0.3](UPGRADING.md) | Updating an existing installation and compatibility changes |
| [Engineering review](ENGINEERING_REVIEW_2026-10-25.md) | Implemented fixes, regression coverage, database-overhead measurements, and limits |
| [Benchmark guide](../benchmarks/README.md) | Fresh retrieval and QA runs, dataset checks, and comparison commands |
| [October benchmark rerun](../benchmarks/BENCHMARK_RERUN_2026-10-25.md) | Latest scores, raw artifacts, API usage, and reproducibility audit |
| [Changelog](../CHANGELOG.md) | Version history |
| [MCP fallback instructions](../CLAUDE.md) | Memory-use guidance for clients without lifecycle integration |

The benchmark reports measure raw-session/window retrieval and answer generation.
The engineering review and product tests cover ingestion, graph integrity,
recovery, and interfaces. These are separate kinds of evidence.

Historical reports retain their original results and link to the current rerun:
[August retrieval](../benchmarks/BENCHMARK_REPORT_2026-08-11.md),
[matched controls](../benchmarks/CONTROLS_ADDENDUM.md), and
[August QA](../benchmarks/QA_ADDENDUM.md).
