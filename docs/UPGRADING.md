# Upgrading to Memoria 0.3

Version 0.3 adds atomic writes, searchable fallback notes, model-free recall,
structured provenance, complete backups, and stricter interface validation.
Existing databases open in place with additive indexes; no export/import or
destructive migration is required. See the [changelog](../CHANGELOG.md) for the
full release and the [engineering review](ENGINEERING_REVIEW_2026-10-25.md) for
compatibility details.

## Update an existing source installation

Stop the hook daemon and other running Memoria HTTP/MCP processes before
updating their installed code:

```bash
memoria daemon stop
```

Preserve every scope database and external archive before the first upgraded
open. On 0.2, stop all writers and copy each database with any remaining
`-wal`/`-shm` sidecars into a separate backup location. Copying only a live
`.db` file can omit committed WAL data. The complete `storage backup` command
is new in 0.3; it is not a prerequisite that an older installation can run.

From your Memoria checkout, with its virtual environment active:

```bash
git switch main
git pull --ff-only
python -m pip install -e .
```

If you use optional hosted extraction, install `'.[llm]'` instead. Reinstall
with the same Python interpreter that your editor/server uses. GUI clients
should reference that interpreter by absolute path as shown in the
[setup instructions](../README.md#setup-by-tool).

Check the global database, then check each project from that project's directory:

```bash
memoria --no-models --scope global storage check
memoria --no-models --scope auto storage check
```

For native integrations, run the relevant `memoria doctor` or
`memoria doctor codex`. If the interpreter or checkout moved, reinstall that
integration with `memoria install claude` or `memoria install codex`. Restart
the editor/server so it loads the new code; hooks start the daemon on demand.

After upgrade, create verified snapshots with `storage backup` and a new
destination per scope. Follow the [storage guide](STORAGE.md) for recovery.

## Changes clients must account for

- Text recall remains the default. Opt into structured results with CLI
  `--json` or HTTP/MCP `format: "json"`; historical recall uses `--as-of` or
  `as_of` respectively.
- HTTP browser access requires explicit `MEMORIA_ALLOWED_ORIGINS`. Binding
  outside loopback requires `MEMORIA_HTTP_TOKEN`; authenticated requests must
  include the bearer token, including health checks.
- HTTP bodies must be JSON objects within the 1 MiB limit. Maintenance flags
  must be booleans, not strings such as `"false"`.
- Archive restores require v1 format and checksum fields. Existing valid v1
  archives remain compatible; incomplete legacy files are rejected.
- CLI environment defaults and explicit database/scope flags are honored,
  including server commands. Confirm the selected scope after upgrading.
- New unstructured notes are searchable by default. Older raw conversations
  that never produced graph facts are not retroactively indexed.
- `--no-models` / `MEMORIA_NO_MODELS=1` uses graph/lexical recall with different
  ranking from neural retrieval. The published neural benchmark scores do not
  describe this mode.
- New database, backup, archive, and pending-state files are private on POSIX.
  Existing file permissions are preserved.

The [October benchmark rerun](../benchmarks/BENCHMARK_RERUN_2026-10-25.md)
confirms unchanged raw-session retrieval metrics. It does not establish that
graph ingestion or model-free recall has the same benchmark accuracy.
