# Memoria

Persistent memory for AI agents. Local-first, no API keys required.

Memoria gives any AI coding tool — Claude Code, Cursor, Codex, or your own agents — a shared long-term memory that persists across conversations. It extracts entities and relationships from conversations, stores them in a knowledge graph, and retrieves them using a three-stage pipeline.

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

The latest audited run is documented in
[the August 2026 benchmark report](benchmarks/BENCHMARK_REPORT_2026-08-11.md).
It uses the official LongMemEval retrieval metric implementation and
apples-to-apples local controls.

| Dataset and scope | Retriever | Recall-All@5 | Recall-All@10 | NDCG-Any@10 |
|---|---:|---:|---:|---:|
| LongMemEval-S official retrieval scope (419) | **Memoria balanced** | **94.3%** | **97.1%** | **0.9478** |
|  | MiniLM dense | 85.4% | 93.8% | 0.8785 |
|  | Flat BM25 | 74.2% | 82.6% | 0.7962 |
| PerLTQA English v2, strict full scope (8,593) | **Memoria windowed** | **84.4%** | — | — |
|  | Windowed BM25 | 80.5% | — | — |
|  | Windowed MiniLM | 71.9% | — | — |
| LoCoMo evidence diagnostic, categories 1–4 (1,536) | **Memoria windowed** | **78.3%** | **87.5%** | **0.8120** |
|  | Flat BM25 | 62.2% | 73.3% | 0.6610 |
|  | Memoria legacy session index | 50.9% | 68.4% | 0.5459 |
|  | MiniLM dense | 41.5% | 55.7% | 0.4612 |

These numbers evaluate benchmark-facing raw-conversation retrieval. They do not
exercise Memoria's conversation ingestion, entity extraction, knowledge-graph
construction, consolidation, or final answer generation. LongMemEval's official
end-to-end score and LoCoMo's headline score are generated-answer metrics, so
the retrieval results above must not be presented as QA accuracy. The LoCoMo
windowed LoCoMo profile was developed after inspecting the public benchmark's
legacy failures, so it is a post-hoc engineering result rather than an unbiased
held-out estimate. PerLTQA English v2 was evaluated afterward as an untouched
external holdout with the retriever frozen; its adapter and data-quality limits
are documented in the report.

### Reproducing benchmarks

```bash
cd benchmarks

# Install benchmark-only comparison dependencies first
pip install -e "..[bench]"

# LongMemEval-S (419 rows in the official retrieval scope)
python download_data.py
python longmemeval_final.py --mode balanced --scope official
python longmemeval_final.py --scope official --retriever flat-bm25
python longmemeval_final.py --scope official --retriever minilm

# LoCoMo session-evidence diagnostic
python locomo_bench.py --mode balanced --profile windowed
python locomo_bench.py --mode balanced --profile legacy
python locomo_bench.py --retriever flat-bm25
python locomo_bench.py --retriever minilm

# PerLTQA English v2 untouched external holdout
python download_perltqa.py
python perltqa_bench.py --retriever memoria --output results_perltqa.json
python perltqa_bench.py --retriever window-bm25 --output results_perltqa_bm25.json
python perltqa_bench.py --retriever window-minilm --output results_perltqa_minilm.json
```

## Installation

### Requirements

- Python 3.10+
- ~500 MB disk for embedding model (downloaded on first use)
- No API keys required for core functionality
- Optional: Anthropic API key for LLM-powered entity extraction (falls back to heuristic extraction without it)

### Install from source

```bash
git clone https://github.com/tomdif/memoria.git
cd memoria
pip install -e .
```

For optional integrations, install the corresponding extra:

```bash
pip install -e ".[mcp]"      # MCP server
pip install -e ".[llm]"      # Anthropic-powered extraction
pip install -e ".[dev]"      # Test suite
```

### Install from PyPI (coming soon)

```bash
pip install memoria-ai
```

## Setup by tool

### Claude Code

```bash
# 1. Add MCP server
pip install -e ".[mcp]"
claude mcp add memoria -- python -m memoria.mcp_server

# 2. Copy the CLAUDE.md to your project (enables automatic, invisible memory)
cp /path/to/memoria/CLAUDE.md .claude/projects/CLAUDE.md
# Or for global (all projects):
cp /path/to/memoria/CLAUDE.md ~/.claude/projects/-global/CLAUDE.md
```

The CLAUDE.md file tells Claude to automatically recall relevant memories at conversation start and store important context as it comes up — completely invisible to the user. Without it, the tools are available but Claude won't use them proactively.

To verify it's working:

```bash
claude
# Ask: "check memoria stats"
# You should see entities and triples accumulating over time
```

### Cursor

1. Add to your Cursor MCP settings (`.cursor/mcp.json` in your project root or `~/.cursor/mcp.json` globally):

```json
{
  "mcpServers": {
    "memoria": {
      "command": "python",
      "args": ["-m", "memoria.mcp_server"],
      "env": {
        "MEMORIA_DB": "~/.memoria/memoria.db"
      }
    }
  }
}
```

2. Add the rules for automatic memory to your Cursor rules file (`.cursor/rules` or `.cursorrules`):

```
Copy the contents of CLAUDE.md from the memoria repo into your rules file.
```

Restart Cursor. Memoria will run invisibly in the background.

### Windsurf

1. Add to your Windsurf MCP configuration (`~/.windsurf/mcp.json`):

```json
{
  "mcpServers": {
    "memoria": {
      "command": "python",
      "args": ["-m", "memoria.mcp_server"],
      "env": {
        "MEMORIA_DB": "~/.memoria/memoria.db"
      }
    }
  }
}
```

2. Add the contents of `CLAUDE.md` from the memoria repo to your Windsurf rules for automatic, invisible memory.

### Cline (VS Code)

1. Add to Cline's MCP settings (VS Code settings → Cline → MCP Servers):

```json
{
  "memoria": {
    "command": "python",
    "args": ["-m", "memoria.mcp_server"],
    "env": {
      "MEMORIA_DB": "~/.memoria/memoria.db"
    }
  }
}
```

2. Add the contents of `CLAUDE.md` from the memoria repo to your Cline custom instructions for automatic, invisible memory.

### Codex CLI / ChatGPT / Custom agents (HTTP API)

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

# Health check
curl http://localhost:7437/health
```

For OpenAI Codex CLI, start the HTTP server and configure Codex to call the endpoints in its system prompt or tool definitions.

### Any tool (CLI)

```bash
# Store a memory
memoria remember "The API rate limit is 1000 req/min, we hit it during the load test"

# Recall memories
memoria recall "rate limit" --mode speed

# Get entity details
memoria entity "API"

# View temporal history
memoria history "rate_limit" --predicate "value"

# Run consolidation (merge duplicates, prune stale memories)
memoria consolidate

# View stats
memoria stats

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
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORIA_DB` | `~/.memoria/memoria.db` | Path to SQLite database |
| `MEMORIA_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `ANTHROPIC_API_KEY` | (none) | Optional, enables LLM entity extraction |

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
- **L1 — Raw store**: Verbatim conversation turns, ground truth
- **L2 — Knowledge graph**: Entities, relations, temporal versioning (supersede-on-conflict)
- **L3 — Spectral clusters**: Emergent topic clusters from graph Laplacian eigenvectors

**Key design choices:**
- **Local-first**: Everything runs on your machine. No cloud, no API calls for core functionality
- **Single SQLite file**: The entire memory state is one file (`~/.memoria/memoria.db`). Back it up, move it, share it
- **Spectral consolidation**: Uses the spectral gap of the knowledge graph to determine search depth and prune low-importance memories
- **History-safe self-cleansing**: Every `consolidate()` call removes self-referencing triples, deduplicates identical facts, merges exact-name duplicate entities, and removes true orphans. Superseded facts remain available to temporal history until `cleanup purge-expired` is explicitly requested
- **Embedding cache**: Session embeddings are cached by content hash, so repeated queries over the same corpus hit memory instead of re-encoding

## MCP Tools

When connected via MCP, memoria exposes these tools:

| Tool | Description |
|------|-------------|
| `memoria_remember` | Store text in long-term memory |
| `memoria_recall` | Retrieve relevant memories (with mode: speed/balanced/quality) |
| `memoria_entity` | Get everything known about an entity |
| `memoria_history` | Temporal history of an entity-predicate pair |
| `memoria_consolidate` | Run memory consolidation (includes automatic self-cleansing) |
| `memoria_stats` | System statistics (entity/triple counts, spectral gap) |
| `memoria_compress` | Compress memory to a token budget using spectral ranking |
| `memoria_budget_report` | Preview compression at each tier (L0/L1/L2/L3) |
| `memoria_cleanup` | Clean up the knowledge graph (delete, merge, deduplicate, purge) |

## License

MIT
