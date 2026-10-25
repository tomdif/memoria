"""Daemon liveness while a neural hook is busy and competing startups occur."""

from pathlib import Path
import tempfile
import threading
import time

import memoria.daemon as daemon
import memoria.hooks as hooks


def test_ping_remains_responsive_and_startup_preserves_live_socket(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    result = []

    def slow_hook(event):
        entered.set()
        assert release.wait(timeout=5)
        return {'done': True}

    monkeypatch.setattr(hooks, 'handle_user_prompt', slow_hook)
    with tempfile.TemporaryDirectory(prefix='memoria-daemon-') as directory:
        # Apply the production short-path logic on platforms with long temp dirs.
        monkeypatch.setenv('MEMORIA_DAEMON_SOCKET', str(Path(directory) / 'memory.sock'))
        path = daemon.socket_path()
        server_thread = threading.Thread(target=daemon.serve, kwargs={'path': path})
        server_thread.start()
        worker = None
        try:
            deadline = time.monotonic() + 3
            while not daemon.daemon_status(path=path) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert daemon.daemon_status(path=path)
            original_inode = path.stat().st_ino
            worker = threading.Thread(target=lambda: result.append(daemon._exchange(
                {'command': 'user-prompt', 'event': {}}, path=path, timeout=5,
            )))
            worker.start()
            assert entered.wait(timeout=2)
            assert daemon.daemon_status(path=path)
            # Competing serve returns immediately because the owner holds a lock.
            daemon.serve(path=path)
            assert path.stat().st_ino == original_inode
            assert daemon.daemon_status(path=path)
        finally:
            release.set()
            if worker:
                worker.join(timeout=5)
            if daemon.daemon_status(path=path):
                daemon._exchange({'command': 'shutdown'}, path=path)
            server_thread.join(timeout=5)
        assert not server_thread.is_alive()
        assert result == [{'ok': True, 'output': {'done': True}}]
        assert not path.exists()
