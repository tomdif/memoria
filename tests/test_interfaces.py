"""Interface-level tests for cleanup and recall dispatch."""

import json
import threading
from http.server import HTTPServer
from urllib.request import Request, urlopen

from memoria.core import Memoria
from memoria.http_server import MemoriaHandler
import memoria.http_server as http_module
import memoria.mcp_server as mcp_module


class FakeMemoria:
    def __init__(self):
        self.calls = []

    def recall_formatted(self, query, top_k=10, mode="balanced"):
        self.calls.append(("recall", query, top_k, mode))
        return f"remembered: {query}"

    def cleanup(self, **kwargs):
        self.calls.append(("cleanup", kwargs))
        return {"purged_triples": 2}

    def graph_stats(self):
        return {"entities": 1}

    def storage_status(self, *, all_scopes=False):
        self.calls.append(("storage", "status", all_scopes))
        return {"database": {"physical_bytes": 1024}}

    def retain_raw_history(self, **kwargs):
        self.calls.append(("storage", "retain", kwargs))
        return {"dry_run": not kwargs.get("apply", False), "candidates": 2}

    def maintain_storage(self, **kwargs):
        self.calls.append(("storage", "maintain", kwargs))
        return {"database": {"vacuumed": kwargs.get("vacuum", False)}}

    def restore_raw_history(self, archive_path, **kwargs):
        self.calls.append(("storage", "restore", archive_path, kwargs))
        return {"dry_run": not kwargs.get("apply", False), "candidates": 2}

    def entity_context(self, name):
        return {"entity": {"name": name}}

    def history(self, name, predicate=None):
        return []


def test_core_cleanup_requires_explicit_history_purge(tmp_path):
    with Memoria(db_path=tmp_path / "memoria.db") as memoria:
        api = memoria.kg.add_entity("API")
        memoria.kg.add_triple(api, "version", object_value="v1")
        memoria.kg.add_triple(api, "version", object_value="v2")

        memoria.consolidate()
        assert len(memoria.history("API", "version")) == 2

        result = memoria.cleanup(action="purge_expired")
        assert result == {"purged_triples": 1}
        assert len(memoria.history("API", "version")) == 1


def test_core_cleanup_entity_and_triple_lifecycle(tmp_path):
    with Memoria(db_path=tmp_path / "cleanup.db") as memoria:
        api = memoria.kg.add_entity("API")
        triple_id, _ = memoria.kg.add_triple(api, "status", object_value="stable")

        assert memoria.cleanup(action="list_entities")["count"] == 1
        assert memoria.cleanup(action="list_triples")["count"] == 1
        assert memoria.cleanup(action="delete_triple", triple_id=triple_id)["deleted"]

        orphans = memoria.cleanup(action="find_orphans")
        assert [entity["name"] for entity in orphans["orphans"]] == ["API"]
        assert memoria.cleanup(action="purge_orphans") == {"purged": 1}
        assert memoria.cleanup(action="list_entities")["count"] == 0


def test_core_cleanup_merge_and_delete_entity(tmp_path):
    with Memoria(db_path=tmp_path / "merge.db") as memoria:
        canonical = memoria.kg.add_entity("Postgres")
        duplicate = memoria.kg.add_entity("Postgres database")
        memoria.kg.add_triple(duplicate, "status", object_value="stable")

        result = memoria.cleanup(
            action="merge_entities",
            entity_name="Postgres database",
            merge_into="Postgres",
        )
        assert result["triples_reassigned"] == 1
        assert memoria.kg.get_entity(duplicate) is None
        assert len(memoria.kg.get_triples(subject_id=canonical)) == 1

        deleted = memoria.cleanup(action="delete_entity", entity_name="Postgres")
        assert deleted["deleted_triples"] == 1
        assert memoria.kg.get_entity(canonical) is None


def test_mcp_cleanup_dispatches_all_arguments(monkeypatch):
    fake = FakeMemoria()
    monkeypatch.setattr(mcp_module, "_memoria", fake)

    result = mcp_module.handle_tool(
        "memoria_cleanup",
        {
            "action": "merge_entities",
            "entity_name": "postgres",
            "entity_id": "source-id",
            "triple_id": "triple-id",
            "merge_into": "Postgres",
        },
    )

    assert result == {"purged_triples": 2}
    assert fake.calls == [(
        "cleanup",
        {
            "action": "merge_entities",
            "entity_name": "postgres",
            "entity_id": "source-id",
            "triple_id": "triple-id",
            "merge_into": "Postgres",
        },
    )]


def test_mcp_storage_retention_is_dry_run_by_default(monkeypatch):
    fake = FakeMemoria()
    monkeypatch.setattr(mcp_module, "_memoria", fake)

    result = mcp_module.handle_tool(
        "memoria_storage",
        {"action": "retain", "older_than_days": 90},
    )

    assert result == {"dry_run": True, "candidates": 2}
    assert fake.calls == [(
        "storage",
        "retain",
        {
            "older_than_days": 90,
            "keep_latest": None,
            "limit": None,
            "archive_path": None,
            "apply": False,
        },
    )]


def test_http_recall_dispatches_mode_and_top_k(monkeypatch):
    fake = FakeMemoria()
    monkeypatch.setattr(http_module, "_memoria", fake)
    server = HTTPServer(("127.0.0.1", 0), MemoriaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        body = json.dumps({
            "query": "database",
            "top_k": 3,
            "mode": "quality",
        }).encode()
        request = Request(
            f"http://127.0.0.1:{server.server_port}/recall",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload == {"status": "ok", "results": "remembered: database"}
    assert fake.calls == [("recall", "database", 3, "quality")]


def test_http_storage_status_supports_all_scopes(monkeypatch):
    fake = FakeMemoria()
    monkeypatch.setattr(http_module, "_memoria", fake)
    server = HTTPServer(("127.0.0.1", 0), MemoriaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/storage?all_scopes=true",
            timeout=2,
        ) as response:
            payload = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload == {"database": {"physical_bytes": 1024}}
    assert fake.calls == [("storage", "status", True)]
