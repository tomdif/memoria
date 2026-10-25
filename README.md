# Memoria

Persistent memory for AI agents. Local-first, no API keys required for the
core path.

Memoria gives Claude Code, Codex, MCP clients, and custom agents shared
long-term memory across conversations. Native lifecycle hooks make recall and
capture deterministic in Claude Code and Codex; MCP, HTTP, and CLI interfaces
cover other tools. Memoria extracts entities and relationships, stores them in
a scoped knowledge graph, and retrieves them with a three-stage pipeline.

Version 0.3 adds atomic memory writes, recall without model downloads,
structured results with provenance, verified database backups, and hardened
HTTP/MCP interfaces. See the [engineering review](docs/ENGINEERING_REVIEW_2026-10-25.md)
for findings, measurements, compatibility changes, and remaining limitations.

Documentation: [API reference](docs/API.md) · [Storage and recovery](docs/STORAGE.md) ·
[Upgrading to 0.3](docs/UPGRADING.md) · [Benchmark guide](benchmarks/README.md) ·
[Changelog](CHANGELOG.md).

## How it works

```
Conversation → Extract entities/relations → Knowledge graph (SQLite)
                                                    ↓
Query → Bi-encoder + BM25 → Cross-encoder rerank → Ranked results
                                                    ↓
                              Spectral consolidation (merge, decay, prune)
```

**Three retrieval passes:**

1. **Graph walk** — Extract entities from query, walk the knowledge graph using spectral-informed depth (the spectral gap of the graph Laplacian determines how far to search)
2. **Scoped vector search** — Bi-encoder similarity + BM25 keyword matching, with adaptive query expansion for multi-topic queries
3. **Cross-encoder rerank** — Rerank top candidates with a cross-encoder, fused with temporal decay and access frequency

Three retrieval modes (`speed`, `balanced`, and `quality`) trade reranking depth
for latency. For long raw conversations, grouped retrieval indexes overlapping
turn windows and collapses them back to unique parent sessions. Explicit
multi-entity comparisons are decomposed into entity-scoped probes, with
parent-level coverage checks and tie-aware reciprocal-rank fusion.

## Benchmarks

The [October 25 rerun](benchmarks/BENCHMARK_RERUN_2026-10-25.md) evaluates the
reliability changes with the original datasets, models, and retrieval settings.
The original protocol is documented in the
[August benchmark report](benchmarks/BENCHMARK_REPORT_2026-08-11.md),
[matched controls and component attribution](benchmarks/CONTROLS_ADDENDUM.md),
and [QA addendum](benchmarks/QA_ADDENDUM.md). LongMemEval scores use its official
metric implementations.

### End-to-end QA accuracy (LongMemEval-S, all 500 questions)

Official LongMemEval judge protocol; identical reader
(`claude-haiku-4-5-20251001`), judge (`claude-sonnet-5`), and official prompts
in both arms. Each run evaluates 500 fresh answers per arm and prompt variant:

| Arm | August plain | October plain | August CoT | October CoT |
|---|---:|---:|---:|---:|
| **Memoria balanced → reader** | 80.8% | **80.2%** | 86.0% | **86.6%** |
| Matched control (hybrid → dual-pass cross-encoder) → reader | 75.6% | 76.4% | 82.8% | 83.8% |

Both systems returned exactly the same ordered top-five evidence lists as in
August. QA differences between runs therefore arise after retrieval; they do
not establish an accuracy gain from the reliability changes. The October
within-run gaps are +3.8 percentage points with the plain reader and +2.8 with
CoT. These are single-run observations, not significance estimates.
One CoT control judgment hit the original token limit and was scored incorrect;
its maximum effect is 0.2 percentage points.

Published LongMemEval-S numbers for context (different readers and judges, so
context rather than a leaderboard: Mem0 66.9%, Zep + gpt-4o 71.2%, LiCoMemory
73.8%, TiMem 76.9%). See the [rerun report](benchmarks/BENCHMARK_RERUN_2026-10-25.md)
for per-ability scores, abstention accuracy, paired answer changes, and limits.

### Retrieval

Every retrieval row below reproduces exactly in the October rerun, including
the full metric and subgroup breakdowns in the raw artifacts.

| Dataset and scope | Retriever | Recall-All@5 | Recall-All@10 | NDCG-Any@10 |
|---|---:|---:|---:|---:|
| LongMemEval-S official retrieval scope (419) | **Memoria balanced** | **94.3%** | **97.1%** | **0.9478** |
|  | Hybrid + dual-pass cross-encoder (matched K=15) | 90.5% | 95.9% | 0.9250 |
|  | Hybrid → cross-encoder | 89.5% | 97.1% | 0.9231 |
|  | MiniLM → cross-encoder | 87.6% | 95.2% | 0.9163 |
|  | MiniLM dense | 85.4% | 93.8% | 0.8785 |
|  | BM25 → cross-encoder | 82.6% | 85.4% | 0.8692 |
|  | Flat BM25 | 74.2% | 82.6% | 0.7962 |

The cross-encoder control rows use the same reranker model and candidate depth
as Memoria's own pipeline, so the +3.8 pp gap over the strongest control is
attributable to Memoria's design rather than commodity reranking. Component
attribution (the 0.40·CE + 0.60·stage-1 score fusion is the dominant term,
+3.1 pp) is in the [controls addendum](benchmarks/CONTROLS_ADDENDUM.md).

| Dataset and scope | Retriever | Recall-All@5 | Recall-All@10 | NDCG-Any@10 |
|---|---:|---:|---:|---:|
| PerLTQA English v2, strict full scope (8,593) | **Memoria windowed** | **84.4%** | — | — |
|  | Windowed BM25 | 80.5% | — | — |
|  | Windowed MiniLM | 71.9% | — | — |
| LoCoMo evidence diagnostic, categories 1–4 (1,536) | **Memoria windowed** | **78.3%** | **87.5%** | **0.8120** |
|  | Flat BM25 | 62.2% | 73.3% | 0.6610 |
|  | Memoria legacy session index | 50.9% | 68.4% | 0.5459 |
|  | MiniLM dense | 41.5% | 55.7% | 0.4612 |

These numbers evaluate benchmark-facing raw-conversation retrieval plus, for
the QA table, official-protocol answer generation. They do not exercise
Memoria's conversation ingestion, entity extraction, knowledge-graph
construction, or consolidation. Retrieval results must not be presented as QA
accuracy — the QA table above is the only generated-answer metric. The LoCoMo
windowed profile was developed after inspecting the public benchmark's
legacy failures, so it is a post-hoc engineering result rather than an unbiased
held-out estimate. PerLTQA English v2 was evaluated afterward as an untouched
external holdout with the retriever frozen; its adapter and data-quality limits
are documented in the report.

### Reproducing benchmarks

Follow the [complete benchmark guide](benchmarks/README.md) for dataset setup,
all 17 retrieval/control configurations, four fresh QA arms, and comparison
commands. Retrieval runs locally without API credentials; QA requires
`ANTHROPIC_API_KEY`. The October four-arm QA run cost an estimated $34.31.
Use a new output directory for each fresh run; existing QA records are resumed.

## Installation

### Requirements

- Python 3.10+
- Default neural models download on first use (about 215 MB in the audited
  environment); Python dependencies need additional disk space
- No API keys required for core functionality
- Optional: Anthropic API key for LLM-powered entity extraction (falls back to heuristic extraction without it)

### Install from source

```bash
git clone https://github.com/tomdif/memoria.git
cd memoria
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The commands above use a POSIX shell. On Windows, activate the environment with
`.venv\Scripts\Activate.ps1` in PowerShell. Lifecycle hooks and the warm daemon
currently target POSIX systems; the tested CI platforms are Linux and macOS.

For optional integrations, install the corresponding extra:

```bash
python -m pip install -e ".[mcp]"      # Optional MCP SDK/CLI tooling
python -m pip install -e ".[llm]"      # Anthropic-powered extraction and QA benchmarks
python -m pip install -e ".[dev]"      # Test suite and lint
```

Source installation is the supported path. The built-in stdio MCP server is
included in the core package; it does not require the optional MCP SDK.
For an existing installation, follow the [upgrade guide](docs/UPGRADING.md).

### Quick start without model downloads

```bash
memoria --no-models --no-llm remember "remember to include reproducible evidence"
memoria --no-models --no-llm recall "reproducible evidence" --json
```

`--no-models` uses graph and lexical retrieval; omit it for neural retrieval.
Set `MEMORIA_NO_MODELS=1` for the same behavior in Python, HTTP, MCP, and hooks.
`--no-llm` or `MEMORIA_NO_LLM=1` disables optional hosted extraction in CLI/MCP.
Unstructured notes are searchable by default, even when extraction finds no facts.

Python callers can use the same API:

```python
from memoria import Memoria

with Memoria("./memory.db", enable_embeddings=False) as memory:
    memory.remember("remember to include reproducible evidence")
    result = memory.recall_json("reproducible evidence", top_k=5)
    print(result["results"][0]["provenance"]["content"])
```

## Setup by tool

Use the Python interpreter where Memoria is installed. GUI clients may not
inherit an activated environment: replace `/absolute/path/to/memoria/.venv/bin/python`
in their configurations with the output of
`python -c 'import sys; print(sys.executable)'`. On Windows this is typically
`.venv\Scripts\python.exe` under your checkout.

### Claude Code

```bash
# Install deterministic lifecycle hooks for all local projects
python -m pip install -e .
memoria install claude
memoria doctor

# Optional: expose the manual inspection and cleanup tools too
claude mcp add memoria -- env MEMORIA_SCOPE=auto \
  "$(python -c 'import sys; print(sys.executable)')" -m memoria.mcp_server
```

The installer adds two Claude Code hooks without replacing your existing
settings:

- `UserPromptSubmit` recalls relevant global and current-project memory and
  injects it as context before Claude handles the prompt.
- `Stop` asynchronously stores durable preferences, decisions, and completed
  work after a turn. Routine tool output, fenced code, and common secret
  patterns are not stored.

Hooks remove the dependency on Claude deciding to call an MCP tool. A backup of
an existing settings file is created on first installation. Use
`memoria install claude --project` for a project-local installation.
The hooks auto-start a user-local daemon over a permission-restricted Unix
socket. It keeps the neural models warm between prompts; inspect it with
`memoria daemon status` and stop it with `memoria daemon stop`.

Memories are isolated structurally. The legacy database is the global scope;
each Git repository gets a separate SQLite database under
`~/.memoria/scopes/`. Worktrees from the same repository share a scope. Recall
searches only the current project plus global user preferences.

To verify manually:

```bash
memoria --scope auto stats
memoria --scope auto recall "what database does this project use" --mode speed
```

### Codex CLI

```bash
# Install deterministic lifecycle hooks for all local projects
memoria install codex
memoria doctor codex
```

Restart Codex after installation, run `/hooks`, and trust the two new Memoria
hooks. Codex requires this one-time review for non-managed command hooks. The
installer preserves existing hooks and creates a first-install backup when
`~/.codex/hooks.json` already exists.

The behavior then matches the Claude Code integration: `UserPromptSubmit`
injects relevant global and repository-scoped memory, while the asynchronous
`Stop` hook captures durable preferences, project decisions, and completed
work. Use `memoria install codex --project` to write `.codex/hooks.json` only
for the current repository; Codex must also trust that project configuration.

### Cursor

1. Add the following to [Cursor's MCP configuration](https://cursor.com/docs/mcp):
   `.cursor/mcp.json` in your project or `~/.cursor/mcp.json` globally.

```json
{
  "mcpServers": {
    "memoria": {
      "command": "/absolute/path/to/memoria/.venv/bin/python",
      "args": ["-m", "memoria.mcp_server"],
      "env": {
        "MEMORIA_DB": "~/.memoria/memoria.db",
        "MEMORIA_SCOPE": "auto"
      }
    }
  }
}
```

2. Create `.cursor/rules/memoria.mdc` with the frontmatter below, then append
   the contents of [CLAUDE.md](CLAUDE.md). Cursor project rules require the
   [`.mdc` format](https://cursor.com/docs/rules).

```markdown
---
description: Use Memoria for durable context across conversations
alwaysApply: true
---
```

Restart Cursor. Because Cursor does not use the Claude lifecycle hooks above,
recall and saving still depend on the agent following these MCP instructions.

### Windsurf / Devin Desktop (Cascade)

1. In Cascade's Actions menu, open the MCP configuration file and add the
   following server. The current [Cascade documentation](https://docs.devin.ai/desktop/cascade/mcp)
   uses `~/.config/devin/mcp_config.json` on macOS/Linux and
   `%APPDATA%\devin\mcp_config.json` on Windows. Opening it through the editor
   avoids guessing the location on older installations. These steps apply to
   Cascade; Devin Local uses its own configuration flow.

```json
{
  "mcpServers": {
    "memoria": {
      "command": "/absolute/path/to/memoria/.venv/bin/python",
      "args": ["-m", "memoria.mcp_server"],
      "env": {
        "MEMORIA_DB": "~/.memoria/memoria.db",
        "MEMORIA_SCOPE": "auto"
      }
    }
  }
}
```

2. Add the contents of [CLAUDE.md](CLAUDE.md) as MCP fallback guidance.

### Cline (VS Code)

1. In Cline's MCP Servers panel, select Configure → Configure MCP Servers.
   Add the entry under `mcpServers`, as described in
   [Cline's MCP documentation](https://github.com/cline/cline/blob/main/docs/mcp/mcp-overview.mdx):

```json
{
  "mcpServers": {
    "memoria": {
      "command": "/absolute/path/to/memoria/.venv/bin/python",
      "args": ["-m", "memoria.mcp_server"],
      "env": {
        "MEMORIA_DB": "~/.memoria/memoria.db",
        "MEMORIA_SCOPE": "auto"
      }
    }
  }
}
```

2. Add the contents of [CLAUDE.md](CLAUDE.md) as MCP fallback guidance.

For MCP clients, `auto` resolves the server process's working directory. If the
client does not start the server in the intended repository, set
`MEMORIA_SCOPE=project:/absolute/path/to/your/project` explicitly.

### Custom agents (HTTP API)

For tools that don't support MCP, memoria provides a REST API:

```bash
# Start the HTTP server (runs on port 7437 by default)
memoria serve-http

# Or with custom port
memoria serve-http --port 8080
```

**API endpoints:**

```bash
# Store a memory
curl -X POST http://localhost:7437/remember \
  -H "Content-Type: application/json" \
  -d '{"text": "We switched from PostgreSQL to SQLite for the config store", "role": "user"}'

# Retrieve memories
curl -X POST http://localhost:7437/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "what database do we use", "top_k": 5, "mode": "speed"}'

# Get entity details
curl http://localhost:7437/entity/SQLite

# Get temporal history
curl http://localhost:7437/history/database

# System stats
curl http://localhost:7437/stats

# Current database, pending-state, and soft-quota usage
curl http://localhost:7437/storage

# Include every project scope and default archive directory
curl 'http://localhost:7437/storage?all_scopes=true'

# Preview raw-provenance retention; apply remains false
curl -X POST http://localhost:7437/storage/retain \
  -H "Content-Type: application/json" \
  -d '{"older_than_days": 180, "keep_latest": 1000}'

# Health check
curl http://localhost:7437/health
```

Codex CLI users should prefer the deterministic hook integration above.
The default binding is `127.0.0.1`. Set `MEMORIA_HTTP_TOKEN` to require bearer
authentication on every data endpoint and health check. Non-loopback bindings
require a token; use TLS at an authenticated reverse proxy for remote access.
Send `Authorization: Bearer <token>` on requests when authentication is enabled.
Browser access is denied unless the exact origin is listed in the comma-separated
`MEMORIA_ALLOWED_ORIGINS` variable. Wildcard CORS is disabled.

Requests must contain JSON objects, use `Content-Type: application/json`, and
fit within 1 MiB. Invalid arguments return HTTP 400; oversized bodies return
413. Maintenance flags such as `apply` accept JSON booleans, not strings.
The HTTP service processes database requests serially and bounds socket reads.

Add `"format": "json"` to `/recall` to receive ranked facts, IDs, scores,
temporal validity, and original provenance. The default text response remains
available. `as_of` accepts a Unix timestamp for historical recall. The MCP
`memoria_recall` tool accepts the same `format` and `as_of` fields.
See the [API reference](docs/API.md) for the complete endpoint list, response
structure, extraction behavior, and interface differences. The HTTP API is a
REST interface; it is not an MCP-over-HTTP endpoint.

### Any tool (CLI)

```bash
# Store a memory
memoria remember "The API rate limit is 1000 req/min, we hit it during the load test"

# Recall memories
memoria recall "rate limit" --mode speed

# Machine-readable results and historical recall
memoria recall "rate limit" --json
memoria recall "rate limit" --as-of 1700000000 --json

# Get entity details
memoria entity "API"

# View temporal history
memoria history "rate_limit" --predicate "value"

# Run consolidation (merge duplicates, prune stale memories)
memoria consolidate

# View stats
memoria stats

# Inspect this scope or the total footprint across every scope
memoria --scope auto storage status
memoria storage status --all-scopes

# Complete, verified backup of the current scope; refuses existing destinations
memoria --scope auto storage backup ./memoria-snapshot.db
memoria --scope auto storage check

# Preview archive-backed raw retention, then explicitly apply it
memoria --scope auto storage retain --older-than-days 180 --keep-latest 1000
memoria --scope auto storage retain --older-than-days 180 --keep-latest 1000 --apply

# Validate or restore a checksummed JSONL archive
memoria --scope auto storage restore ~/.memoria/scopes/archives/archive.jsonl
memoria --scope auto storage restore ~/.memoria/scopes/archives/archive.jsonl --apply

# Truncate the WAL; add --vacuum to physically reclaim free DB pages
memoria --scope auto storage compact
memoria --scope auto storage compact --vacuum

# Compress memory to fit a token budget
memoria compress --budget 500

# Clean up: list, delete, merge, find issues
memoria cleanup list-entities
memoria cleanup list-triples
memoria cleanup find-duplicates
memoria cleanup find-orphans
memoria cleanup purge-orphans
memoria cleanup purge-expired  # permanently deletes superseded history
memoria cleanup delete-entity --name "old entity"
memoria cleanup delete-triple --id "triple-uuid"
memoria cleanup merge --name "duplicate" --into "canonical"

# Isolate a command to the current repository
memoria --scope auto remember "This project uses PostgreSQL"
memoria --scope auto recall "database"
```

CLI, MCP, and HTTP retain the legacy `global` default for backward
compatibility. Set `MEMORIA_SCOPE=auto` for repository isolation. Claude Code
and Codex lifecycle hooks always use the current project plus global user
preferences.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORIA_DB` | `~/.memoria/memoria.db` | Path to SQLite database |
| `MEMORIA_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `MEMORIA_NO_MODELS` | (none) | Set to `1`, `true`, or `yes` for graph/lexical retrieval without local neural models |
| `MEMORIA_HTTP_TOKEN` | (none) | Optional bearer token; required when HTTP binds outside loopback |
| `MEMORIA_ALLOWED_ORIGINS` | (none) | Exact comma-separated browser origins allowed to access HTTP |
| `MEMORIA_SCOPE` | interface-dependent | `global`, `auto`, or `project:/path` |
| `MEMORIA_HOOK_MODE` | `speed` | Retrieval mode used by Claude Code and Codex hooks |
| `MEMORIA_HOOK_STATE_DIR` | `~/.memoria/hook-state` | Pending-turn state for asynchronous saving |
| `MEMORIA_PENDING_TTL_DAYS` | `7` | Automatic lifetime for abandoned temporary hook prompts; negative disables cleanup |
| `MEMORIA_MAX_DB_MB` | (none) | Per-scope soft quota: reports and returns warnings but keeps accepting writes |
| `MEMORIA_HARD_MAX_DB_MB` | (none) | Per-scope hard quota: refuses new memory writes at the boundary without deleting existing data |
| `MEMORIA_DAEMON_SOCKET` | `~/.memoria/memoria.sock` | Local hook-daemon socket |
| `MEMORIA_DAEMON_LOG` | `~/.memoria/daemon.log` | Hook-daemon log path |
| `MEMORIA_NO_LLM` | (none) | Set to `1`, `true`, or `yes` to disable optional CLI/MCP LLM extraction |
| `MEMORIA_HOOK_DEBUG` | (none) | Set to `1` to report hook errors to stderr |
| `ANTHROPIC_API_KEY` | (none) | Enables optional CLI/MCP LLM extraction when the `llm` extra is installed; required for QA benchmarks |

CLI commands honor `MEMORIA_DB`, `MEMORIA_SCOPE`, and `MEMORIA_MODEL`;
explicit `--db` and `--scope` flags take precedence, including server commands.

## Storage, RAM, and retention

The warm hook daemon holds the embedding and reranking models. Model memory is
roughly fixed, but retrieval buffers grow with the eligible graph and corpus.
Actual process memory depends on platform, dependencies, and loaded models.
The two default model downloads currently require about 215 MB; a complete
Python environment can be substantially larger. Measure the actual database
and temporary-state footprint with:

```bash
memoria storage status --all-scopes
```

SQLite data grows only when Memoria accepts durable memory. Growth varies with
text length and extracted graph structure, so Memoria reports bytes rather
than assuming a fixed cost per turn. `MEMORIA_MAX_DB_MB` supplies an observable
soft threshold. `MEMORIA_HARD_MAX_DB_MB` is opt-in and refuses new writes after
the current scope reaches the threshold; neither setting automatically deletes
existing memory.

Retention is deliberately explicit and recoverable:

1. `storage retain` is a dry-run unless `--apply` is present.
2. Applied retention writes old raw provenance to a permission-restricted,
   checksummed JSONL archive before deleting the local raw rows.
3. Knowledge-graph facts remain searchable, and their source IDs reconnect if
   the archive is restored.
4. `storage restore` validates checksums and is also dry-run by default.
5. `storage compact --vacuum` reclaims pages only after retention or cleanup.

Raw retention does not delete active graph facts. Use `consolidate` for graph
decay/deduplication and the explicit cleanup commands for unwanted facts or
superseded history. Abandoned hook prompts are temporary rather than durable
memory and are automatically removed after seven days. See
[the storage operations guide](docs/STORAGE.md) for recovery semantics,
capacity planning, and safe production procedures.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Interfaces                      │
│  MCP Server (stdio)  │  HTTP API  │  CLI         │
└────────────┬─────────┴─────┬──────┴──────┬───────┘
             │               │             │
         ┌───▼───────────────▼─────────────▼───┐
         │              Memoria Core            │
         │  remember() → extract → store        │
         │  recall()   → 3-pass retrieve        │
         │  consolidate() → spectral prune      │
         └──┬──────┬──────┬──────┬─────────────┘
            │      │      │      │
    ┌───────▼┐ ┌───▼───┐ ┌▼─────┐ ┌▼──────────┐
    │Raw     │ │Knowledge│ │Vector│ │Spectral   │
    │Store   │ │Graph    │ │Index │ │Analysis   │
    │(L1)    │ │(L2)     │ │      │ │           │
    └────────┘ └────────┘ └──────┘ └───────────┘
                    │
              SQLite (WAL mode)
```

**Layers:**

- **L1 — Raw store**: Verbatim provenance, retained locally or moved to checksummed JSONL archives
- **L2 — Knowledge graph**: Entities, relations, temporal versioning (supersede-on-conflict)
- **L3 — Spectral clusters**: Emergent topic clusters from graph Laplacian eigenvectors

**Key design choices:**

- **Atomic writes**: Raw provenance, entity updates, facts, and supersession commit together; failed writes and failed consolidation roll back
- **Local-first**: Everything runs on your machine. No cloud, no API calls for core functionality
- **Scoped SQLite files**: Global memory stays in `~/.memoria/memoria.db`; project memory uses one isolated database per Git repository under `~/.memoria/scopes/`
- **Bounded-operation controls**: Per-scope soft/hard quotas, all-scope usage reporting, archive-first retention, restore validation, WAL checkpointing, and explicit vacuuming
- **Spectral consolidation**: Uses the spectral gap of the knowledge graph to determine search depth and prune low-importance memories
- **History-safe self-cleansing**: Every `consolidate()` call removes self-referencing triples, deduplicates identical facts, merges exact-name duplicate entities, and removes true orphans. Superseded facts remain available to temporal history until `cleanup purge-expired` is explicitly requested
- **Bounded embedding cache**: Session embeddings use a 4,096-entry LRU cache with deduplicated batch encoding; evicted entries can be recomputed

## MCP Tools

When connected via MCP, memoria exposes these tools:

| Tool | Description |
|------|-------------|
| `memoria_remember` | Store text in long-term memory |
| `memoria_recall` | Retrieve relevant memories; supports `mode`, `format: json`, and historical `as_of` |
| `memoria_entity` | Get everything known about an entity |
| `memoria_history` | Temporal history of an entity-predicate pair |
| `memoria_consolidate` | Run memory consolidation (includes automatic self-cleansing) |
| `memoria_stats` | System statistics (entity/triple counts, spectral gap) |
| `memoria_storage` | Usage/quota status, dry-run retention and restore, compaction, and temporary-state cleanup |
| `memoria_compress` | Compress memory to a token budget using spectral ranking |
| `memoria_budget_report` | Preview compression at each tier (L0/L1/L2/L3) |
| `memoria_cleanup` | Clean up the knowledge graph (delete, merge, deduplicate, purge) |

## Development and verification

```bash
python -m pip install -e '.[dev]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
ruff check .
```

The 0.3 suite has 194 tests and needs neither model downloads nor API keys.
[CI](https://github.com/tomdif/memoria/actions/workflows/ci.yml) checks Linux
with Python 3.10/3.13, macOS with Python 3.13, and wheel/sdist packaging.
See the [documentation index](docs/README.md) for operational guides and
the separate product-reliability and benchmark evidence.

## License

MIT
