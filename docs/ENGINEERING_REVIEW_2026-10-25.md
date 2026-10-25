# Memoria 0.3 engineering review

Reviewed the product ingestion and retrieval paths, SQLite graph and raw store,
consolidation, hook daemon, CLI, HTTP, MCP, packaging, and existing tests against
baseline commit `918b39a`. The original 84 tests passed before changes.

The highest-value issues were data integrity and integration reliability. The
existing benchmark reports evaluate raw-session retrieval and QA, so they do
not establish correctness of the product write path.

## Findings and implemented changes

| Finding | Resulting behavior |
|---|---|
| Raw rows, entities, and facts committed independently; later failures left partial memory | One transaction commits raw provenance, facts, and supersession. Savepoints compose with caller transactions; provider/model work runs before the write lock. Consolidation and multi-step cleanup also roll back on failure. |
| Entity lookup used prefix matches during ingestion | Ingestion resolves exact names; lookup escapes literal `%` and `_` and ranks exact matches first. Concurrent writers serialize lookup-and-create. |
| Entity-to-literal changes and empty values could evade contradiction detection | Object identity and literal values are compared together. Epoch-zero timestamps are preserved; supersession closes at the replacement's validity boundary. Out-of-order replacements that would invalidate the current fact's interval are rejected. |
| Text without extracted facts was effectively absent from recall | New unstructured input gets a complete searchable note by default, linked to its raw provenance. |
| Disabling embeddings still loaded a cross-encoder; empty databases could load models | `enable_embeddings=False`, `--no-models`, or `MEMORIA_NO_MODELS=1` uses graph/lexical retrieval. Empty recall returns before model initialization. |
| Per-entity embedding calls and per-fact SQL reads amplified work | Unique names are embedded in one batch; entity names and provenance are fetched in bounded SQL batches. Case-insensitive lookup and source-reference indexes are added idempotently. |
| Session embedding cache grew without a bound | A configurable LRU defaults to 4,096 entries, deduplicates batch inputs, and copies cached vectors so slices cannot retain entire batch allocations. |
| Degenerate clustering could pass an invalid probability distribution to NumPy | Identical points and two-node graphs are handled; automatic clustering requests at most 32 eigenvectors. |
| Slow inference blocked daemon pings, allowing competing starts to replace its socket | Management requests run independently; neural hook work remains serialized. An OS file lock protects the socket for the server lifetime. |
| HTTP allowed wildcard browser access and accepted malformed bodies/flags | Exact-origin allowlist, optional bearer auth, required auth outside loopback, rebinding protection, bounded reads, strict JSON objects, and client error responses. String `"false"` cannot apply retention. Internal error details stay in server logs. |
| MCP malformed input could stop the service; notifications received replies | Bounded framing, parse/argument errors, notification handling, ping, protocol-version negotiation, and tool error indicators. Model output is redirected away from protocol stdout. |
| CLI ignored documented database/scope environment variables and server flags | Environment defaults and explicit flags work consistently; invalid operations exit nonzero with concise errors. |
| Recall returned only human-readable strings at integration boundaries | Opt-in structured JSON includes fact IDs, names, scores, source channels, validity, scope, and raw provenance. CLI/HTTP/MCP expose historical recall. |
| JSONL archives were the only recovery mechanism; removing a checksum bypassed validation | Complete SQLite backups include committed WAL data and graph state, verify integrity before publication, and never overwrite destinations. Archive format and checksum are mandatory. `storage check` verifies pages, foreign keys, and FTS consistency. |
| New database and pending-prompt files inherited broad default permissions | Newly created database, backup, archive, and pending-prompt files are private on POSIX. Existing file permissions are preserved. |
| No repository CI | Linux/Python 3.10 and 3.13 plus macOS/Python 3.13 tests, Python error checks, and wheel/sdist validation are configured. |

## Measured database overhead

The [reproducible probe](../benchmarks/product_reliability_probe.py) ingests one
synthetic extraction containing 1,000 entities and 1,000 facts, then recalls ten
facts. It uses temporary SQLite files, fixed extraction output, and a fixed
reranker. Three runs per revision used the same local Python 3.13 environment.

| Measurement | Baseline | Updated |
|---|---:|---:|
| Commits per ingestion | 2,001 | 1 |
| SELECT queries for recall document loading | 1,002 | 3 |
| Median ingestion wall time | 336.441 ms | 159.151 ms |

[Raw measurements](reliability-probe-2026-10-25.json) accompany the probe.
This is a database-overhead measurement, not an inference benchmark or a claim
of improved retrieval accuracy. Timing depends on filesystem and hardware.

Reproduce against two local source trees:

```bash
python benchmarks/product_reliability_probe.py --source-tree /path/to/baseline
python benchmarks/product_reliability_probe.py --source-tree .
```

## Verification

- All 193 tests pass on Python 3.10 and Python 3.13, up from 84 baseline tests.
- Failure injection checks that raw rows, FTS entries, graph facts, and
  supersession roll back together and that the connection remains usable.
- Concurrent independent SQLite writers create one shared entity and preserve
  every additive fact. Nested transactions do not commit their caller's work.
- Real HTTP sockets exercise malformed bodies, limits, authentication, browser
  origins, rebinding, maintenance flags, and sanitized errors.
- CLI, MCP, and HTTP subprocess tests exercise complete remember/recall flows,
  structured provenance, environment/flag handling, and clean shutdown.
- Backup recovery and deliberately damaged FTS/archives verify failure paths.
- A blocked hook verifies daemon health checks remain responsive and a competing
  startup preserves the existing socket.
- The suite runs without model downloads or API keys. Local Python 3.10 and 3.13
  were both checked. A separate smoke test loaded the cached MiniLM embedder and
  cross-encoder, correctly retrieving a database fact and an unstructured note.
- Source and wheel distributions build successfully; Python error lint passes.

Run verification with:

```bash
pip install -e '.[dev]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
ruff check memoria tests
```

## Compatibility and remaining limits

- Existing databases open in place with additional indexes; no data migration or
  destructive cleanup is required. New fallback notes do not retroactively
  index legacy conversations that have no graph facts.
- The existing text recall format remains the default. JSON is opt-in.
- HTTP browser clients need an explicit allowed origin. Non-loopback deployments
  need `MEMORIA_HTTP_TOKEN`; remote use still needs TLS at the deployment boundary.
- Maintenance flags must be real booleans. Invalid extraction shapes and invalid
  values fail before storage. Empty/incomplete extracted statements are skipped.
- Archives missing format/checksum fields are rejected. Existing checked v1
  archives remain supported. A checksum detects accidental changes; it is not
  a cryptographic signature against someone who can rewrite the archive.
- A complete backup covers one scope. Back up other scopes and external JSONL
  archives separately; see [storage operations](STORAGE.md).
- Model-free recall is lexical and graph-based, with different ranking from the
  neural modes. It is not represented by the published benchmark scores.
- Retrieval still scans the eligible graph facts and rebuilds lexical scoring
  per query. Large-corpus indexing and model-version-aware re-embedding remain
  future work. Cache eviction trades bounded RAM for possible re-encoding.
- Heuristic extraction can lose semantic detail. Structured recall exposes the
  raw source when available, but retrieval quality and contradiction semantics
  still need evaluation on real ingestion workloads.
- The spectral radius is a retrieval heuristic, not a semantic completeness
  guarantee. The existing LongMemEval/LoCoMo/PerLTQA scores were not rerun or
  revised as part of this reliability work.
- The HTTP service remains serial for SQLite ownership; the hook daemon handles
  management requests concurrently while serializing neural hook operations.
