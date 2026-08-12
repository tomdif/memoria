"""User-local daemon for low-latency Claude Code and Codex memory hooks."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


MAX_MESSAGE_BYTES = 2 * 1024 * 1024


def socket_path() -> Path:
    requested = Path(
        os.environ.get("MEMORIA_DAEMON_SOCKET", "~/.memoria/memoria.sock")
    ).expanduser()
    # macOS limits AF_UNIX paths to roughly 104 bytes. Test runners and deeply
    # nested homes can exceed that, so fall back to a stable short temp path.
    if len(os.fsencode(str(requested))) >= 100:
        digest = hashlib.sha256(str(requested).encode("utf-8")).hexdigest()[:20]
        return Path(tempfile.gettempdir()) / f"memoria-{digest}" / "m.sock"
    return requested


def log_path() -> Path:
    return Path(
        os.environ.get("MEMORIA_DAEMON_LOG", "~/.memoria/daemon.log")
    ).expanduser()


def _exchange(
    request: dict[str, Any],
    *,
    timeout: float = 3.0,
    path: str | Path | None = None,
) -> dict[str, Any]:
    target = str(Path(path).expanduser() if path else socket_path())
    payload = json.dumps(request, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("daemon request is too large")

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(target)
        client.sendall(payload)
        chunks = bytearray()
        while len(chunks) <= MAX_MESSAGE_BYTES:
            block = client.recv(65536)
            if not block:
                break
            chunks.extend(block)
            if b"\n" in block:
                break
    if not chunks:
        raise ConnectionError("daemon closed the connection without a response")
    response = json.loads(bytes(chunks).split(b"\n", 1)[0])
    if not isinstance(response, dict):
        raise ValueError("invalid daemon response")
    return response


def daemon_status(*, path: str | Path | None = None) -> bool:
    try:
        return _exchange({"command": "ping"}, timeout=0.5, path=path).get("ok") is True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def start_daemon(
    *,
    path: str | Path | None = None,
    wait_seconds: float = 5.0,
) -> bool:
    """Start a detached daemon if one is not already responsive."""
    target = Path(path).expanduser() if path else socket_path()
    if daemon_status(path=target):
        return True

    parent_existed = target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        try:
            target.parent.chmod(0o700)
        except OSError:
            pass
    log = log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MEMORIA_DAEMON_SOCKET"] = str(target)

    with log.open("ab") as log_stream:
        subprocess.Popen(
            [sys.executable, "-m", "memoria.daemon", "serve"],
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=log_stream,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if daemon_status(path=target):
            return True
        time.sleep(0.05)
    return False


def stop_daemon(*, path: str | Path | None = None) -> bool:
    target = Path(path).expanduser() if path else socket_path()
    if not daemon_status(path=target):
        return False
    try:
        _exchange({"command": "shutdown"}, timeout=2.0, path=target)
    except OSError:
        pass
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not daemon_status(path=target):
            return True
        time.sleep(0.05)
    return not daemon_status(path=target)


def hook_request(
    command: str,
    event: dict[str, Any],
    *,
    timeout: float = 50.0,
) -> dict[str, Any] | None:
    """Send a hook event to the daemon, starting it on demand."""
    if command not in {"user-prompt", "stop"}:
        raise ValueError(f"unsupported hook command: {command}")
    if not start_daemon():
        raise ConnectionError("Memoria daemon did not start")
    response = _exchange(
        {"command": command, "event": event},
        timeout=timeout,
    )
    if response.get("ok") is not True:
        raise RuntimeError(str(response.get("error") or "daemon request failed"))
    output = response.get("output")
    return output if isinstance(output, dict) else None


class _DaemonServer(socketserver.UnixStreamServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        super().server_bind()
        try:
            Path(self.server_address).chmod(0o600)
        except OSError:
            pass


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
        if len(raw) > MAX_MESSAGE_BYTES:
            self._respond({"ok": False, "error": "request is too large"})
            return
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            command = request.get("command")
            if command == "ping":
                response = {"ok": True, "pid": os.getpid()}
            elif command == "shutdown":
                response = {"ok": True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            elif command in {"user-prompt", "stop"}:
                event = request.get("event")
                if not isinstance(event, dict):
                    raise ValueError("hook request requires an event object")
                # Import lazily so ping and management commands stay fast.
                from .hooks import handle_stop, handle_user_prompt

                output = (
                    handle_user_prompt(event)
                    if command == "user-prompt"
                    else handle_stop(event)
                )
                response = {"ok": True, "output": output}
            else:
                raise ValueError(f"unknown daemon command: {command}")
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        self._respond(response)

    def _respond(self, response: dict[str, Any]) -> None:
        try:
            self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(*, path: str | Path | None = None) -> None:
    target = Path(path).expanduser() if path else socket_path()
    parent_existed = target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not parent_existed:
        try:
            target.parent.chmod(0o700)
        except OSError:
            pass

    if target.exists():
        if daemon_status(path=target):
            return
        target.unlink()

    server = _DaemonServer(str(target), _RequestHandler)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if command == "serve":
        serve()
    elif command == "start":
        raise SystemExit(0 if start_daemon() else 1)
    elif command == "status":
        print("running" if daemon_status() else "stopped")
        raise SystemExit(0 if daemon_status() else 1)
    elif command == "stop":
        raise SystemExit(0 if stop_daemon() else 1)
    else:
        print("Usage: python -m memoria.daemon start|status|stop|serve", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
