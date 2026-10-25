# Changelog

## 0.3.0 — 2026-10-25

- Made ingestion, consolidation, and multi-step graph cleanup atomic, with
  nested savepoints and serialized concurrent entity creation.
- Fixed entity prefix collisions, wildcard lookup, mixed entity/literal
  contradictions, epoch-zero validity, and malformed extraction handling.
- Added searchable fallback notes by default, model-free graph/lexical recall,
  structured provenance, and historical recall across CLI, HTTP, and MCP.
- Batched embedding and SQL work, bounded the embedding cache, and handled
  empty lexical corpora and degenerate spectral clustering.
- Kept daemon health checks responsive during inference and protected startup
  with an OS lock so competing processes cannot replace a live socket.
- Hardened HTTP authentication, browser origins, request sizes, input types,
  error responses, and cleanup. Non-loopback binding now requires a token.
- Hardened MCP framing, notifications, argument validation, version negotiation,
  and machine-readable stdout. CLI environment defaults and server flags work
  consistently and errors return nonzero exit codes.
- Added complete verified SQLite backups and storage integrity checks; archive
  format/checksum validation is mandatory. New private data files use `0600`.
- Added cross-platform CI and regression coverage for failure recovery,
  concurrency, interface security, backups, and process-level integrations.

See [the engineering review](docs/ENGINEERING_REVIEW_2026-10-25.md) for measured
database overhead, compatibility details, verification, and remaining limits.

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
