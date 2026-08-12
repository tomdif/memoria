"""
MCP server — exposes memoria as tools for Claude Code, Cursor, etc.

Run: python -m memoria.mcp_server
Or configure in Claude Code settings.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from .core import Memoria
from . import __version__


# Lazy-init the memoria instance
_memoria: Memoria | None = None


def _make_llm_call():
    """Create an LLM call function using the Anthropic SDK for entity extraction."""
    try:
        import anthropic
        client = anthropic.Anthropic()

        def call(prompt: str) -> str:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call
    except (ImportError, Exception):
        return None


def get_memoria() -> Memoria:
    global _memoria
    if _memoria is None:
        db_path = os.environ.get("MEMORIA_DB", "~/.memoria/memoria.db")
        model = os.environ.get("MEMORIA_MODEL", "all-MiniLM-L6-v2")
        use_llm = os.environ.get("MEMORIA_NO_LLM", "").lower() not in ("1", "true", "yes")
        llm_call = _make_llm_call() if use_llm else None
        scope = os.environ.get("MEMORIA_SCOPE", "global")
        _memoria = Memoria(
            db_path=db_path,
            model_name=model,
            llm_call=llm_call,
            scope=scope,
        )
    return _memoria


# --- Tool definitions ---

TOOLS = [
    {
        "name": "memoria_remember",
        "description": "Store a conversation turn or piece of information in long-term memory. "
                       "Extracts entities and relationships automatically.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to remember"},
                "role": {"type": "string", "enum": ["user", "assistant", "system"], "default": "user"},
                "session_id": {"type": "string", "description": "Optional session identifier"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "memoria_recall",
        "description": "Retrieve memories relevant to a query. Uses three-pass retrieval: "
                       "graph walk (spectral-informed depth), scoped vector search, temporal reranking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "top_k": {"type": "integer", "default": 10, "description": "Max results"},
                "mode": {"type": "string", "enum": ["speed", "balanced", "quality"],
                         "default": "balanced",
                         "description": "Retrieval mode: speed (16q/s), balanced (12q/s), quality (5q/s)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memoria_entity",
        "description": "Get everything known about a specific entity — current facts, "
                       "history (including superseded values), and relationships.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name to look up"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "memoria_history",
        "description": "Get the full temporal history of an entity-predicate pair, "
                       "including what was true at each point in time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "Entity name"},
                "predicate": {"type": "string", "description": "Optional predicate to filter"},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "memoria_consolidate",
        "description": "Run memory consolidation: merge repeated facts, apply spectral decay, "
                       "prune low-confidence memories, recompute clusters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prune_threshold": {"type": "number", "default": 0.05},
                "half_life_days": {"type": "number", "default": 30.0},
            },
        },
    },
    {
        "name": "memoria_stats",
        "description": "Get system statistics: entity/triple counts, spectral gap, "
                       "screening radius, cluster count.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "memoria_storage",
        "description": "Inspect or explicitly maintain Memoria storage. Retention is "
                       "archive-first and dry-run by default; graph facts remain available.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "retain", "restore", "maintain"],
                    "default": "status",
                },
                "older_than_days": {"type": "number"},
                "keep_latest": {"type": "integer"},
                "limit": {"type": "integer"},
                "archive_path": {"type": "string"},
                "apply": {"type": "boolean", "default": False},
                "vacuum": {"type": "boolean", "default": False},
                "stale_pending_days": {"type": "number", "default": 7},
                "apply_stale_cleanup": {"type": "boolean", "default": False},
                "all_scopes": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "memoria_compress",
        "description": "Compress full memory state into a token budget using spectral ranking. "
                       "Returns the most structurally important facts that fit within the budget. "
                       "Tiers: L0 (≤50 tokens, identity), L1 (≤200, key facts), "
                       "L2 (≤2000, cluster summaries), L3 (>2000, full detail).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "budget_tokens": {
                    "type": "integer",
                    "default": 200,
                    "description": "Maximum tokens for the compressed output",
                },
            },
        },
    },
    {
        "name": "memoria_budget_report",
        "description": "Show what memory content you'd get at each compression tier "
                       "(L0/L1/L2/L3), with previews and compression ratios.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "memoria_cleanup",
        "description": "Clean up the knowledge graph: list/delete entities and triples, "
                       "merge duplicates, find true orphans, or explicitly purge temporal history. "
                       "Actions: list_entities, list_triples, delete_entity, delete_triple, "
                       "merge_entities, find_duplicates, find_orphans, purge_orphans, purge_expired.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_entities", "list_triples", "delete_entity",
                             "delete_triple", "merge_entities", "find_duplicates",
                             "find_orphans", "purge_orphans", "purge_expired"],
                    "description": "The cleanup action to perform",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Entity name (for delete_entity, merge_entities source)",
                },
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID (for delete_entity by ID)",
                },
                "triple_id": {
                    "type": "string",
                    "description": "Triple ID (for delete_triple)",
                },
                "merge_into": {
                    "type": "string",
                    "description": "Target entity name to merge into (for merge_entities)",
                },
            },
            "required": ["action"],
        },
    },
]


def handle_tool(name: str, arguments: dict[str, Any]) -> dict:
    """Execute a tool call and return the result."""
    m = get_memoria()

    if name == "memoria_remember":
        result = m.remember(
            text=arguments["text"],
            role=arguments.get("role", "user"),
            session_id=arguments.get("session_id"),
        )
        return {"status": "ok", **result}

    elif name == "memoria_recall":
        result = m.recall_formatted(
            query=arguments["query"],
            top_k=arguments.get("top_k", 10),
            mode=arguments.get("mode", "balanced"),
        )
        return {"status": "ok", "results": result}

    elif name == "memoria_entity":
        result = m.entity_context(arguments["name"])
        return result

    elif name == "memoria_history":
        result = m.history(
            entity_name=arguments["entity"],
            predicate=arguments.get("predicate"),
        )
        return {"history": result, "count": len(result)}

    elif name == "memoria_consolidate":
        result = m.consolidate(
            prune_threshold=arguments.get("prune_threshold", 0.05),
            half_life_days=arguments.get("half_life_days", 30.0),
        )
        return {"status": "ok", **result}

    elif name == "memoria_stats":
        return m.graph_stats()

    elif name == "memoria_storage":
        action = arguments.get("action", "status")
        if action == "status":
            return m.storage_status(all_scopes=arguments.get("all_scopes", False))
        if action == "retain":
            return m.retain_raw_history(
                older_than_days=arguments.get("older_than_days"),
                keep_latest=arguments.get("keep_latest"),
                limit=arguments.get("limit"),
                archive_path=arguments.get("archive_path"),
                apply=arguments.get("apply", False),
            )
        if action == "restore":
            archive_path = arguments.get("archive_path")
            if not archive_path:
                return {"error": "archive_path is required for restore"}
            return m.restore_raw_history(
                archive_path,
                limit=arguments.get("limit"),
                apply=arguments.get("apply", False),
            )
        if action == "maintain":
            return m.maintain_storage(
                vacuum=arguments.get("vacuum", False),
                stale_pending_days=arguments.get("stale_pending_days", 7),
                apply_stale_cleanup=arguments.get("apply_stale_cleanup", False),
            )
        return {"error": f"Unknown storage action: {action}"}

    elif name == "memoria_compress":
        result = m.compress(budget_tokens=arguments.get("budget_tokens", 200))
        return {
            "tier": result.tier,
            "text": result.text,
            "token_estimate": result.token_estimate,
            "entities_included": result.entities_included,
            "entities_total": result.entities_total,
            "compression_ratio": result.compression_ratio,
        }

    elif name == "memoria_budget_report":
        return m.budget_report()

    elif name == "memoria_cleanup":
        result = m.cleanup(
            action=arguments["action"],
            entity_name=arguments.get("entity_name"),
            entity_id=arguments.get("entity_id"),
            triple_id=arguments.get("triple_id"),
            merge_into=arguments.get("merge_into"),
        )
        return result

    else:
        return {"error": f"Unknown tool: {name}"}


# --- MCP Protocol (stdio JSON-RPC) ---

def run_mcp_server():
    """Run as MCP server over stdio (JSON-RPC 2.0)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method", "")
        req_id = request.get("id")

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "memoria", "version": __version__},
                },
            }

        elif method == "notifications/initialized":
            continue

        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS},
            }

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            try:
                result = handle_tool(tool_name, tool_args)
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                    },
                }
            except Exception as e:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                        "isError": True,
                    },
                }

        else:
            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_mcp_server()
