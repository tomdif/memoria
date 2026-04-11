# Memoria

Persistent memory for AI agents. Local-first, no API keys required.

Memoria gives any AI coding tool — Claude Code, Cursor, Codex, or your own agents — a shared long-term memory that persists across conversations. It extracts entities and relationships from conversations, stores them in a knowledge graph, and retrieves them using a three-stage pipeline that achieves **95.2% Recall@5** on LongMemEval (500 questions).

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

**Three retrieval modes** to trade speed for accuracy:

| Mode | LongMemEval R@5 | LoCoMo R@5 | Throughput |
|------|----------------|-----------|------------|
| `speed` | 94.0% | 63.2% | 15+ q/s |
| `balanced` | 93.6% | 63.2% | 11+ q/s |
| `quality` | 95.0% | 63.2% | 5+ q/s |

All benchmarks on Apple Silicon (M-series), single-threaded, no GPU.

## Benchmarks

### LongMemEval (500 questions)

The [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) benchmark tests retrieval across ~53 conversation sessions per question, covering six categories of conversational memory.

**Overall results (balanced mode, 30% BM25 fusion):**

| Metric | Score |
|--------|-------|
| Recall@5 | **93.6%** |
| Recall@10 | **97.0%** |
| NDCG@10 | **0.9557** |

**Breakdown by question type:**

| Category | R@5 | R@10 | NDCG@10 | n |
|----------|-----|------|---------|---|
| Single-session (user) | 100.0% | 100.0% | 0.971 | 70 |
| Single-session (assistant) | 100.0% | 100.0% | 0.987 | 56 |
| Single-session (preference) | 96.7% | 96.7% | 0.863 | 30 |
| Knowledge update | 100.0% | 100.0% | 0.985 | 78 |
| Multi-session | 95.5% | 98.5% | 0.976 | 133 |
| Temporal reasoning | 87.2% | 94.0% | 0.929 | 133 |

**Speed vs. quality tradeoff (all 500 questions):**

| Config | R@5 | R@10 | NDCG@10 | Time | q/s |
|--------|-----|------|---------|------|-----|
| top-10 single-pass | 93.2% | 95.4% | 0.9416 | 24.9s | 20.1 |
| top-15 single-pass | 93.6% | 96.8% | 0.9426 | 25.4s | 19.7 |
| **top-20 single-pass (speed)** | **94.0%** | **97.4%** | **0.9428** | **30.4s** | **16.4** |
| top-25 single-pass | 94.0% | 97.2% | 0.9417 | 35.2s | 14.2 |
| **top-15 dual-pass (balanced)** | **94.6%** | **97.0%** | **0.9557** | **40.3s** | **12.4** |
| **top-20 dual-pass (quality)** | **95.0%** | **97.6%** | **0.9570** | **102.0s** | **4.9** |
| top-50 dual-pass | 95.2% | 97.8% | 0.9579 | 133.1s | 3.8 |

Theoretical ceiling: 99.4% R@5 (3 questions require >5 answer sessions).

### LoCoMo (1,982 questions)

The [LoCoMo](https://github.com/snap-research/locomo) benchmark tests multi-hop reasoning across 10 long conversations (19-32 sessions each, 400-600 dialog turns). This is a significantly harder benchmark — questions require finding evidence scattered across multiple sessions in much longer conversation histories.

**Overall results (balanced mode, 30% BM25 fusion):**

| Metric | Score |
|--------|-------|
| Recall@5 | **63.2%** |
| Recall@10 | **80.2%** |

**Breakdown by category:**

| Category | R@5 | R@10 | n |
|----------|-----|------|---|
| Single-fact temporal | 77.3% | 88.2% | 321 |
| Open-ended | 71.5% | 89.7% | 446 |
| Multi-fact temporal | 69.3% | 87.6% | 841 |
| Multi-fact | 38.0% | 53.3% | 92 |
| Single-fact | 23.8% | 42.9% | 282 |

LoCoMo's single-fact questions are adversarially hard — they require finding one specific line in a 20+ session conversation. Temporal questions are easier because date context narrows the search space.

### Reproducing benchmarks

```bash
cd benchmarks

# LongMemEval (500 questions, ~2-5 min depending on mode)
python download_data.py
python longmemeval_final.py

# LoCoMo (1,982 questions, ~2 min)
python locomo_bench.py

# Full head-to-head comparison
python headtohead_bench.py --max 500
```

## Installation

### Requirements

- Python 3.9+
- ~500 MB disk for embedding model (downloaded on first use)
- No API keys required for core functionality
- Optional: Anthropic API key for LLM-powered entity extraction (falls back to heuristic extraction without it)

### Install from source

```bash
git clone https://github.com/tomdif/memoria.git
cd memoria
pip install -e .
```

### Install from PyPI (coming soon)

```bash
pip install memoria-ai
```

## Setup by tool

### Claude Code

```bash
# 1. Add MCP server
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
memoria cleanup purge-expired
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
- **Self-cleansing**: Every `consolidate()` call automatically removes self-referencing triples, deduplicates identical facts, merges exact-name duplicate entities, removes orphan entities, and purges expired data — no manual cleanup needed
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
