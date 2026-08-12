# Changelog

## 0.2.0 — 2026-08-12

- Added native, deterministic Claude Code and Codex lifecycle hooks with
  repository-scoped memory and a warm local daemon.
- Added `memoria install codex`, `memoria doctor codex`, idempotent hook
  installation, and Codex-compatible hook output.
- Added current-scope and all-scope storage reporting, soft and hard per-scope
  quotas, archive-first raw retention, checksummed restore, WAL checkpointing,
  optional VACUUM, and stale temporary-hook cleanup.
- Added MCP and HTTP storage-management surfaces; mutating retention and restore
  operations remain dry-run unless explicitly applied.
- Added audited LongMemEval-S, LoCoMo, and PerLTQA retrieval benchmarks and
  clarified the distinction between evidence retrieval and end-to-end QA.

## 0.1.0

- Initial local-first knowledge graph, spectral consolidation, hybrid
  retrieval, CLI, MCP server, and HTTP API.
