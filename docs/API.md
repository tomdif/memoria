# API reference

This guide describes Memoria 0.3. See the [README](../README.md) for installation
and [storage guide](STORAGE.md) for backup and maintenance procedures.

## Python

```python
from memoria import Memoria

with Memoria('./memory.db', scope='global', enable_embeddings=False) as memory:
    memory.remember('remember to include reproducible evidence')
    response = memory.recall_json('reproducible evidence', top_k=5)
    for result in response['results']:
        print(result['fact']['id'], result['score'], result['provenance'])
    assert memory.check_storage()['ok']
```

`enable_embeddings=False` disables both local embedding and reranking models.
Omitting it honors `MEMORIA_NO_MODELS`; otherwise neural retrieval is enabled.
Python uses heuristic extraction unless an explicit `llm_call(prompt) -> str`
callback is supplied. An API key alone does not enable hosted extraction in the
Python API, HTTP service, or lifecycle hooks. CLI and MCP can create an Anthropic
callback when the `llm` extra and API key are available; disable that with
`--no-llm` or `MEMORIA_NO_LLM=1`.

| Method | Result / behavior |
|---|---|
| `remember(text, role='user', session_id=None, metadata=None)` | Atomically stores raw provenance and derived graph data; returns ingestion counts |
| `recall(query, top_k=20, mode='balanced', as_of=None)` | Returns a `RetrievalResponse` with ranked results and retrieval diagnostics |
| `recall_formatted(query, top_k=20, mode='balanced', as_of=None)` | Human-readable recall text |
| `recall_json(query, top_k=20, mode='balanced', as_of=None)` | JSON-compatible facts, provenance, and diagnostics |
| `entity_context(entity_name)` / `history(entity_name, predicate=None)` | Entity information and temporal history |
| `consolidate(**kwargs)` | Applies graph merging, decay, and pruning in a transaction |
| `compress(budget_tokens=200)` / `budget_report()` | Compressed memory or budget previews |
| `graph_stats()` / `storage_status(all_scopes=False)` | Graph or physical-storage statistics |
| `backup(destination)` / `check_storage()` | Verified complete SQLite snapshot or integrity result |

Use the context manager or call `close()` when finished. A `Memoria` instance
owns a SQLite connection; do not share it across threads. Separate instances
can write to the same database with SQLite's locking and transaction semantics.

## Structured recall and time

Python `recall_json`, CLI `recall --json`, and HTTP/MCP `format: "json"` expose
the same result data. Each entry contains:

- `fact`: stored fact fields, including IDs, `subject_name`, `object_name`,
  source reference, and validity timestamps;
- `score`, `source`, and `depth`: ranking information and retrieval channel;
- `provenance`: the original raw record, or null if that record is unavailable
  or has been archived out of the active database.

The response also includes `count`, `mode`, `scope`, `query_entities`,
`passes_used`, `screening_depth`, and `local_spectral_gap`. HTTP/MCP recall
adds `status: "ok"`. Text recall remains the default at those interfaces.

`as_of` is a finite Unix timestamp in seconds; the CLI spells it `--as-of`.
It selects facts valid at that point in time, including superseded facts whose
validity interval contains it. It does not reconstruct an entire historical
database snapshot. `cleanup purge-expired` permanently removes superseded
history and can therefore change historical recall.

`top_k` must be an integer from 0 to 1,000. Its default is 20 in Python and 10
in CLI, HTTP, and MCP. Modes are `speed`, `balanced`, and `quality`; their
latency depends on corpus size, hardware, and model availability.

## Scope and model configuration

CLI, HTTP, and MCP default to global memory. `--scope auto` or
`MEMORIA_SCOPE=auto` selects a repository scope from the process working
directory. For a service launched elsewhere, use
`MEMORIA_SCOPE=project:/absolute/path/to/project`. Python callers pass `scope`
explicitly and can supply `scope_cwd` for automatic resolution.

Lifecycle hooks recall the current project plus global user preferences.
Direct CLI/HTTP/MCP queries operate on their configured scope. Global data
defaults to `~/.memoria/memoria.db`; projects have separate files under its
sibling `scopes/` directory. Git worktrees belonging to the same repository
share a scope.

`MEMORIA_DB` and `MEMORIA_MODEL` configure CLI/HTTP/MCP and hook runtimes.
Python callers pass `db_path` and `model_name` explicitly. Existing embeddings
are not automatically migrated when the model changes; avoid changing models
on an existing database without a separate migration plan.

## HTTP

```bash
memoria --no-models --scope auto serve-http --host 127.0.0.1 --port 7437
```

This is a REST API, separate from the stdio MCP server. Requests with bodies
must use `Content-Type: application/json` and a JSON object. Bodies are limited
to 1 MiB. Invalid arguments return 400, unknown endpoints 404, unsupported
content types 415, and oversized bodies 413. Internal failures return a
sanitized error while details go to the server log.

| Method / path | Fields or behavior |
|---|---|
| `GET /health` (or `/`) | Health and package version |
| `GET /stats` | Graph statistics |
| `GET /entity/{name}` | URL-encoded entity name |
| `GET /history/{name}?predicate=...` | History; optional predicate filter |
| `GET /storage?all_scopes=true` | Storage usage; defaults to the configured scope |
| `POST /remember` | Required `text`; optional `role`, `session_id`, `metadata` |
| `POST /recall` | Required `query`; optional `top_k`, `mode`, `format`, `as_of` |
| `POST /consolidate` | Optional `prune_threshold` (0.05), `half_life_days` (30); applies immediately |
| `POST /compress` | Optional `budget_tokens` (200) |
| `POST /storage/retain` | `older_than_days`, `keep_latest`, `limit`, `archive_path`, `apply` |
| `POST /storage/restore` | Required `archive_path`; optional `limit`, `apply` |
| `POST /storage/maintain` | WAL checkpoint; optional `vacuum`, `stale_pending_days`, `apply_stale_cleanup` |

Example structured recall:

```bash
curl http://127.0.0.1:7437/recall \
  -H 'Content-Type: application/json' \
  -d '{"query":"database","top_k":5,"mode":"balanced","format":"json"}'
```

When `MEMORIA_HTTP_TOKEN` is set, every endpoint, including health, requires
`Authorization: Bearer <token>`. Non-loopback bindings require a token. Browser
origins must match an entry in `MEMORIA_ALLOWED_ORIGINS` exactly; wildcards are
not supported. Remote deployments need TLS at their proxy boundary.

`apply`, `vacuum`, and `apply_stale_cleanup` require actual JSON booleans.
Retention and restore default to previews. Maintenance checkpoints the WAL
even when its optional flags are false. The service handles database requests
serially. Complete backups and integrity checks use CLI/Python, not HTTP.

## MCP

Launch `python -m memoria.mcp_server` with the installed interpreter. It uses
stdio JSON-RPC; model/log output stays off protocol stdout. The server exposes
the ten tools listed in the [README](../README.md#mcp-tools). `tools/list`
returns their current argument schemas. `memoria_storage` supports `status`,
`retain`, `restore`, and `maintain`; it does not expose backup/check actions.

Example arguments for `memoria_recall`:

```json
{"query":"database","top_k":5,"mode":"balanced","format":"json"}
```

Tool results are JSON-encoded text within MCP `content`; tool failures set
`isError`. Malformed requests receive protocol errors and notifications receive
no reply. For clients without lifecycle hooks, use the
[MCP fallback guidance](../CLAUDE.md) to request recall and capture explicitly.
