# Storage operations

Memoria 0.2 adds observable, bounded storage operations without silently
deleting durable memory. Limits apply per SQLite scope: the global database and
each Git repository have separate files.

## What consumes space

- The warm hook daemon keeps neural models in RAM. Its footprint is mostly
  fixed and does not grow with conversation count.
- Downloaded embedding and reranking models are cached outside the Memoria
  database. The current defaults occupy about 215 MB, though dependency and
  model releases can change this.
- Accepted durable memories add raw provenance, full-text index entries,
  entities, embeddings, triples, and indexes to SQLite.
- Project isolation creates one database per Git repository under
  `~/.memoria/scopes/`.
- WAL and shared-memory sidecars can temporarily make the physical footprint
  larger than the logical database.

Use current measurements instead of a fixed bytes-per-turn estimate:

```bash
# Current scope
memoria --scope auto storage status

# Global DB, every project DB, and archives in the default locations
memoria storage status --all-scopes
```

The report separates the main database, WAL, shared-memory file, active logical
pages, reclaimable pages, raw text, entity embeddings, pending hook state, and
configured quota state.

## Quotas

Both quotas are optional and apply to each scope independently:

```bash
# Warn in status and remember responses above 512 MB, but keep writing
export MEMORIA_MAX_DB_MB=512

# Refuse new writes when a scope has reached 1 GB
export MEMORIA_HARD_MAX_DB_MB=1024
```

The hard quota is checked before a new raw record is written. It never deletes
or overwrites existing memory. A single accepted record can place a database
slightly over the exact threshold because the check uses its pre-write size.
Hook integrations fail open for the agent conversation if memory storage is
unavailable; inspect the daemon log and `storage status` operationally.

## Archive-backed retention

Retention affects old raw provenance, not active graph facts. Combine an age
boundary with a minimum recent-record floor:

```bash
# Preview only
memoria --scope auto storage retain \
  --older-than-days 180 \
  --keep-latest 1000

# Archive eligible rows, fsync the archive, then delete local raw rows
memoria --scope auto storage retain \
  --older-than-days 180 \
  --keep-latest 1000 \
  --apply
```

When both boundaries are present, a record must be older than the age cutoff
and outside the newest-record floor. `--limit` bounds a maintenance batch.
Without `--archive`, global archives are written under `~/.memoria/archives/`
and project archives under `~/.memoria/scopes/archives/`.

Each JSONL record contains the original ID, timestamp, role, content, session,
metadata, format identifier, archive time, and SHA-256 checksum. New archives
are mode `0600` where the operating system supports POSIX permissions. Archives
contain plaintext memory and should receive the same protection as the source
database.

Applied retention completes the archive write and filesystem sync before the
database transaction deletes rows. Full-text deletion triggers keep search in
sync. Graph triples keep their source IDs and remain retrievable even while the
raw record is offline.

## Restore

Validation is non-mutating by default:

```bash
memoria --scope auto storage restore /path/to/archive.jsonl
memoria --scope auto storage restore /path/to/archive.jsonl --apply
```

Restore rejects symbolic links, malformed records, and checksum mismatches.
Existing conversation IDs are left unchanged and reported as skipped. Restored
IDs reconnect graph provenance automatically.

## Reclaiming physical pages

Archiving rows makes pages reusable by SQLite but may not immediately shrink
the database file. Checkpoint first, then vacuum during a maintenance window:

```bash
# Truncate the WAL and preview stale temporary hook files
memoria --scope auto storage compact

# Rebuild SQLite and reclaim free pages
memoria --scope auto storage compact --vacuum
```

VACUUM needs temporary free disk space roughly comparable to the database.
Memoria checks available capacity before starting. It can take an exclusive
write lock, so stop high-traffic MCP/HTTP writers or the hook daemon first for a
predictable production maintenance window.

## Graph retention

Raw archival deliberately preserves derived graph memory. To reduce graph
growth:

```bash
memoria --scope auto consolidate
memoria --scope auto cleanup find-duplicates
memoria --scope auto cleanup find-orphans
memoria --scope auto cleanup purge-expired
```

`purge-expired`, entity deletion, and triple deletion permanently change graph
history. Review their output and backups before applying them. Consolidation
preserves superseded temporal history unless it is explicitly purged.

## Temporary hook state

Claude Code and Codex hooks briefly store pending prompts until the matching
`Stop` event. Abandoned JSON files are removed automatically after seven days.
Set `MEMORIA_PENDING_TTL_DAYS` to change the lifetime or a negative value to
disable automatic cleanup.

Manual maintenance previews stale files by default:

```bash
memoria storage compact --stale-pending-days 7
memoria storage compact --stale-pending-days 7 --apply-stale-cleanup
```

This cleanup touches only pending hook-state JSON, never durable SQLite memory
or archives.
