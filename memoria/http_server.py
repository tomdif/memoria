"""Bounded JSON HTTP API for local agents, with optional bearer authentication.

Set MEMORIA_HTTP_TOKEN for authenticated access (required for non-loopback
bindings). Browser clients must also be listed in MEMORIA_ALLOWED_ORIGINS.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import TCPServer
from urllib.parse import urlparse, parse_qs, unquote

from .core import Memoria
from . import __version__, validation


_memoria: Memoria | None = None
MAX_BODY_BYTES = 1024 * 1024
REQUEST_TIMEOUT = 10
_LOG = logging.getLogger(__name__)


def get_memoria() -> Memoria:
    global _memoria
    if _memoria is None:
        _memoria = Memoria(
            db_path=os.environ.get("MEMORIA_DB", "~/.memoria/memoria.db"),
            model_name=os.environ.get("MEMORIA_MODEL", "all-MiniLM-L6-v2"),
            scope=os.environ.get("MEMORIA_SCOPE", "global"),
        )
    return _memoria


def _origins() -> set[str]:
    return {value.strip() for value in os.environ.get("MEMORIA_ALLOWED_ORIGINS", "").split(",") if value.strip()}


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class HTTPError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class MemoriaHTTPServer(HTTPServer):
    """Bind locally without HTTPServer's blocking reverse-DNS lookup.

    The API does not use a canonical DNS server name. Resolving one can stall
    startup on offline machines or hosts with slow/misconfigured DNS.
    """

    def server_bind(self):
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


class MemoriaHandler(BaseHTTPRequestHandler):
    """Serial requests keep the SQLite connection on its owning thread."""

    def setup(self):
        super().setup()
        self.connection.settimeout(REQUEST_TIMEOUT)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, default=str, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin")
        if origin and origin in _origins():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        if self.command == "OPTIONS" and status == 200:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _guard(self):
        origin = self.headers.get("Origin")
        if origin and origin not in _origins():
            raise HTTPError("Browser origin is not allowed", 403)
        # Block DNS rebinding against the unauthenticated loopback service.
        bind_host = self.server.server_address[0]
        host = urlparse("//" + self.headers.get("Host", "")).hostname
        if _loopback(bind_host) and (host is None or not _loopback(host)):
            raise HTTPError("Host is not allowed", 403)
        token = os.environ.get("MEMORIA_HTTP_TOKEN", "")
        if token and self.command != "OPTIONS":
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
                raise HTTPError("Authentication required", 401)

    def _read_body(self) -> dict:
        if self.headers.get("Transfer-Encoding"):
            raise HTTPError("Transfer-Encoding is not supported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) > 1:
            raise HTTPError("Ambiguous Content-Length")
        try:
            length = int(lengths[0]) if lengths else 0
        except ValueError as exc:
            raise HTTPError("Invalid Content-Length") from exc
        if length < 0:
            raise HTTPError("Invalid Content-Length")
        if length > MAX_BODY_BYTES:
            raise HTTPError("Request body is too large", 413)
        if not length:
            return {}
        if self.headers.get_content_type() != "application/json":
            raise HTTPError("Content-Type must be application/json", 415)
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise HTTPError("Incomplete request body")
        try:
            body = json.loads(raw, parse_constant=_invalid_constant)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise HTTPError("Request body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise HTTPError("Request body must be a JSON object")
        return body

    def _run(self, dispatch):
        try:
            self._guard()
            self._send_json(dispatch())
        except HTTPError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except (ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, 400)
        except (socket.timeout, TimeoutError):
            self._send_json({"error": "Request timed out"}, 408)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:
            _LOG.exception("Memoria HTTP request failed")
            self._send_json({"error": "Internal server error"}, 500)

    def do_OPTIONS(self):
        self._run(lambda: {})

    def do_GET(self):
        self._run(self._get)

    def do_POST(self):
        self._run(self._post)

    def _get(self):
        parsed = urlparse(self.path)
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        qs = parse_qs(parsed.query)
        if not parts or parts == ["health"]:
            return {"status": "ok", "version": __version__}
        if parts == ["stats"]:
            return get_memoria().graph_stats()
        if parts == ["storage"]:
            value = qs.get("all_scopes", ["false"])[0].casefold()
            if value not in {"true", "false", "1", "0", "yes", "no"}:
                raise HTTPError("all_scopes must be true or false")
            return get_memoria().storage_status(all_scopes=value in {"true", "1", "yes"})
        if len(parts) == 2 and parts[0] == "entity":
            return get_memoria().entity_context(parts[1])
        if len(parts) == 2 and parts[0] == "history":
            result = get_memoria().history(parts[1], predicate=qs.get("predicate", [None])[0])
            return {"history": result, "count": len(result)}
        raise HTTPError("Unknown endpoint", 404)

    def _post(self):
        path = urlparse(self.path).path.rstrip("/")
        if path not in {"/remember", "/recall", "/consolidate", "/compress", "/storage/retain", "/storage/restore", "/storage/maintain"}:
            raise HTTPError("Unknown endpoint", 404)
        body = self._read_body()
        if path == "/remember":
            validation.text(body.get("text"), "text")
            result = get_memoria().remember(
                text=body["text"], role=body.get("role", "user"),
                session_id=body.get("session_id"), metadata=body.get("metadata"),
            )
            return {"status": "ok", **result}
        if path == "/recall":
            validation.text(body.get("query"), "query")
            validation.integer(body.get("top_k", 10), "top_k", maximum=1000)
            if body.get("mode", "balanced") not in {"speed", "balanced", "quality"}:
                raise HTTPError("mode must be speed, balanced, or quality")
            if body.get("format", "text") not in {"text", "json"}:
                raise HTTPError("format must be text or json")
            kwargs = {"query": body["query"], "top_k": body.get("top_k", 10), "mode": body.get("mode", "balanced")}
            if "as_of" in body:
                validation.number(body["as_of"], "as_of")
                kwargs["as_of"] = body["as_of"]
            if body.get("format") == "json":
                return {"status": "ok", **get_memoria().recall_json(**kwargs)}
            return {"status": "ok", "results": get_memoria().recall_formatted(**kwargs)}
        if path == "/consolidate":
            result = get_memoria().consolidate(
                prune_threshold=body.get("prune_threshold", 0.05),
                half_life_days=body.get("half_life_days", 30.0),
            )
            return {"status": "ok", **result}
        if path == "/compress":
            result = get_memoria().compress(budget_tokens=body.get("budget_tokens", 200))
            return {key: getattr(result, key) for key in (
                "tier", "text", "token_estimate", "entities_included", "entities_total", "compression_ratio",
            )}
        if path == "/storage/retain":
            validation.boolean(body.get("apply", False), "apply")
            return get_memoria().retain_raw_history(
                older_than_days=body.get("older_than_days"), keep_latest=body.get("keep_latest"),
                limit=body.get("limit"), archive_path=body.get("archive_path"), apply=body.get("apply", False),
            )
        if path == "/storage/restore":
            validation.text(body.get("archive_path"), "archive_path")
            validation.boolean(body.get("apply", False), "apply")
            return get_memoria().restore_raw_history(
                body["archive_path"], limit=body.get("limit"), apply=body.get("apply", False),
            )
        validation.boolean(body.get("vacuum", False), "vacuum")
        validation.boolean(body.get("apply_stale_cleanup", False), "apply_stale_cleanup")
        return get_memoria().maintain_storage(
            vacuum=body.get("vacuum", False), stale_pending_days=body.get("stale_pending_days", 7),
            apply_stale_cleanup=body.get("apply_stale_cleanup", False),
        )

    def log_message(self, format, *args):
        _LOG.info(format, *args)


def run_http_server(port: int = 7437, host: str = "127.0.0.1"):
    """Start the HTTP API and release sockets and SQLite on shutdown."""
    global _memoria
    if not _loopback(host) and not os.environ.get("MEMORIA_HTTP_TOKEN"):
        raise ValueError("Set MEMORIA_HTTP_TOKEN before binding to a non-loopback address")
    server = MemoriaHTTPServer((host, port), MemoriaHandler)
    print(f"Memoria HTTP API running on http://{host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if _memoria is not None:
            _memoria.close()
            _memoria = None
