"""
HTTP REST API — makes memoria accessible to any AI tool.

Runs alongside or instead of the MCP server. Any tool that can make
HTTP requests (Codex CLI, ChatGPT plugins, custom agents, curl) can
use memoria through this API.

Usage:
  memoria serve-http                    # default port 7437
  memoria serve-http --port 8080
  MEMORIA_DB=~/my.db memoria serve-http

Endpoints:
  POST /remember    — store a memory
  POST /recall      — retrieve memories
  GET  /entity/:name — entity details
  GET  /history/:name — temporal history
  POST /consolidate — run consolidation
  GET  /stats       — system stats
  POST /compress    — compress to token budget
  GET  /health      — health check
"""

from __future__ import annotations

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

from .core import Memoria


_memoria: Memoria | None = None


def get_memoria() -> Memoria:
    global _memoria
    if _memoria is None:
        db_path = os.environ.get("MEMORIA_DB", "~/.memoria/memoria.db")
        model = os.environ.get("MEMORIA_MODEL", "all-MiniLM-L6-v2")
        _memoria = Memoria(db_path=db_path, model_name=model)
    return _memoria


class MemoriaHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Memoria REST API."""

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _path_parts(self) -> list[str]:
        return [p for p in urlparse(self.path).path.split("/") if p]

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        parts = self._path_parts()
        try:
            m = get_memoria()

            if not parts or parts[0] == "health":
                self._send_json({"status": "ok", "version": "0.1.0"})

            elif parts[0] == "stats":
                self._send_json(m.graph_stats())

            elif parts[0] == "entity" and len(parts) >= 2:
                name = unquote(parts[1])
                self._send_json(m.entity_context(name))

            elif parts[0] == "history" and len(parts) >= 2:
                name = unquote(parts[1])
                qs = parse_qs(urlparse(self.path).query)
                predicate = qs.get("predicate", [None])[0]
                result = m.history(name, predicate=predicate)
                self._send_json({"history": result, "count": len(result)})

            else:
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def do_POST(self):
        parts = self._path_parts()
        try:
            m = get_memoria()
            body = self._read_body()

            if parts[0] == "remember":
                if "text" not in body:
                    self._send_json({"error": "Missing required field: text"}, 400)
                    return
                result = m.remember(
                    text=body["text"],
                    role=body.get("role", "user"),
                    session_id=body.get("session_id"),
                    metadata=body.get("metadata"),
                )
                self._send_json({"status": "ok", **result})

            elif parts[0] == "recall":
                if "query" not in body:
                    self._send_json({"error": "Missing required field: query"}, 400)
                    return
                result = m.recall_formatted(
                    query=body["query"],
                    top_k=body.get("top_k", 10),
                    mode=body.get("mode", "balanced"),
                )
                self._send_json({"status": "ok", "results": result})

            elif parts[0] == "consolidate":
                result = m.consolidate(
                    prune_threshold=body.get("prune_threshold", 0.05),
                    half_life_days=body.get("half_life_days", 30.0),
                )
                self._send_json({"status": "ok", **result})

            elif parts[0] == "compress":
                result = m.compress(budget_tokens=body.get("budget_tokens", 200))
                self._send_json({
                    "tier": result.tier,
                    "text": result.text,
                    "token_estimate": result.token_estimate,
                    "entities_included": result.entities_included,
                    "entities_total": result.entities_total,
                    "compression_ratio": result.compression_ratio,
                })

            else:
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def log_message(self, format, *args):
        sys.stderr.write(f"[memoria] {args[0]} {args[1]} {args[2]}\n")


def run_http_server(port: int = 7437, host: str = "127.0.0.1"):
    """Start the HTTP REST API server."""
    server = HTTPServer((host, port), MemoriaHandler)
    print(f"Memoria HTTP API running on http://{host}:{port}")
    print(f"  POST /remember    — store a memory")
    print(f"  POST /recall      — retrieve memories")
    print(f"  GET  /entity/:name — entity details")
    print(f"  GET  /stats       — system stats")
    print(f"  GET  /health      — health check")
    print(f"\nPress Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
