"""Storage capacity, retention, archival, and compaction tests."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from memoria.core import Memoria
from memoria.maintenance import clean_stale_pending


def _seed_raw_history(memoria: Memoria) -> list[str]:
    ids = [
        memoria.store.append("oldest retained provenance", session_id="s1"),
        memoria.store.append("older retained provenance", session_id="s2"),
        memoria.store.append("newest retained provenance", session_id="s3"),
    ]
    now = time.time()
    memoria.db.executemany(
        "UPDATE conversations SET timestamp = ? WHERE id = ?",
        [
            (now - 30 * 86400, ids[0]),
            (now - 20 * 86400, ids[1]),
            (now - 1 * 86400, ids[2]),
        ],
    )
    entity = memoria.kg.add_entity("Retention test")
    memoria.kg.add_triple(
        entity,
        "records",
        object_value="durable graph fact",
        source_ref=ids[0],
    )
    memoria.db.commit()
    return ids


def test_retention_is_dry_run_then_archive_first(tmp_path):
    archive = tmp_path / "archive" / "history.jsonl"
    with Memoria(
        db_path=tmp_path / "memoria.db",
        enable_embeddings=False,
    ) as memoria:
        ids = _seed_raw_history(memoria)

        preview = memoria.retain_raw_history(
            older_than_days=10,
            keep_latest=1,
            archive_path=archive,
        )
        assert preview["dry_run"] is True
        assert preview["candidates"] == 2
        assert memoria.store.count() == 3
        assert not archive.exists()

        result = memoria.retain_raw_history(
            older_than_days=10,
            keep_latest=1,
            archive_path=archive,
            apply=True,
        )

        assert result["archived"] == 2
        assert result["deleted"] == 2
        assert memoria.store.count() == 1
        assert memoria.store.get(ids[2]) is not None
        assert memoria.store.search_fts("oldest") == []
        triples = memoria.kg.get_triples(active_only=False)
        assert len(triples) == 1
        assert triples[0]["object_value"] == "durable graph fact"
        assert triples[0]["source_ref"] == ids[0]

        restore_preview = memoria.restore_raw_history(archive)
        assert restore_preview["dry_run"] is True
        assert restore_preview["candidates"] == 2
        assert memoria.store.count() == 1

        restored = memoria.restore_raw_history(archive, apply=True)
        assert restored["restored"] == 2
        assert restored["skipped_existing"] == 0
        assert memoria.store.count() == 3
        assert memoria.store.search_fts("oldest")[0]["id"] == ids[0]

    records = [json.loads(line) for line in archive.read_text().splitlines()]
    assert [record["id"] for record in records] == ids[:2]
    assert all(record["archived_at"] for record in records)
    assert all(record["archive_format"] == "memoria.raw.v1" for record in records)
    assert all(len(record["checksum_sha256"]) == 64 for record in records)
    assert archive.stat().st_mode & 0o777 == 0o600


def test_retention_limit_and_validation(tmp_path):
    with Memoria(
        db_path=tmp_path / "memoria.db",
        enable_embeddings=False,
    ) as memoria:
        _seed_raw_history(memoria)
        result = memoria.retain_raw_history(keep_latest=0, limit=1)
        assert result["candidates"] == 1

        try:
            memoria.retain_raw_history()
        except ValueError as exc:
            assert "older_than" in str(exc)
        else:
            raise AssertionError("retention without a boundary must fail")


def test_storage_status_and_soft_quota_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIA_MAX_DB_MB", "0.000001")
    with Memoria(
        db_path=tmp_path / "memoria.db",
        enable_embeddings=False,
    ) as memoria:
        result = memoria.remember(
            "This project uses Postgres",
            ensure_retrievable=True,
        )
        status = memoria.storage_status(all_scopes=True)

    database = status["database"]
    assert database["physical_bytes"] > 0
    assert database["quota_exceeded"] is True
    assert database["quota_percent"] > 100
    assert result["storage_warning"]["quota_bytes"] == database["quota_bytes"]
    assert status["all_scopes"]["database_count"] == 1
    assert status["all_scopes"]["total_managed_bytes"] >= database["physical_bytes"]


def test_hard_quota_refuses_new_writes_without_deleting_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIA_HARD_MAX_DB_MB", "0.000001")
    with Memoria(
        db_path=tmp_path / "memoria.db",
        enable_embeddings=False,
    ) as memoria:
        try:
            memoria.remember("This write should be refused")
        except RuntimeError as exc:
            assert "MEMORIA_HARD_MAX_DB_MB" in str(exc)
        else:
            raise AssertionError("hard quota must refuse writes")
        assert memoria.store.count() == 0


def test_compaction_and_stale_pending_cleanup(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    stale = state_dir / "stale.json"
    fresh = state_dir / "fresh.json"
    stale.write_text("{}")
    fresh.write_text("{}")
    old = time.time() - 10 * 86400
    os.utime(stale, (old, old))
    monkeypatch.setenv("MEMORIA_HOOK_STATE_DIR", str(state_dir))

    preview = clean_stale_pending(state_dir, older_than_days=7)
    assert preview["dry_run"] is True
    assert preview["files"] == 1
    assert stale.exists()

    with Memoria(
        db_path=tmp_path / "memoria.db",
        enable_embeddings=False,
    ) as memoria:
        memoria.store.append("temporary row")
        result = memoria.maintain_storage(
            vacuum=True,
            stale_pending_days=7,
            apply_stale_cleanup=True,
        )

    assert result["database"]["vacuumed"] is True
    assert result["stale_pending"]["files"] == 1
    assert not stale.exists()
    assert fresh.exists()


def test_archive_rejects_symbolic_link(tmp_path):
    target = tmp_path / "real.jsonl"
    target.write_text("")
    link = tmp_path / "archive.jsonl"
    link.symlink_to(target)
    with Memoria(
        db_path=tmp_path / "memoria.db",
        enable_embeddings=False,
    ) as memoria:
        memoria.store.append("old row")
        try:
            memoria.retain_raw_history(
                keep_latest=0,
                archive_path=link,
                apply=True,
            )
        except ValueError as exc:
            assert "regular file" in str(exc)
        else:
            raise AssertionError("symbolic-link archive must be rejected")
        assert memoria.store.count() == 1


def test_restore_rejects_modified_archive(tmp_path):
    archive = tmp_path / "history.jsonl"
    with Memoria(
        db_path=tmp_path / "memoria.db",
        enable_embeddings=False,
    ) as memoria:
        memoria.store.append("original content")
        memoria.retain_raw_history(
            keep_latest=0,
            archive_path=archive,
            apply=True,
        )
        record = json.loads(archive.read_text())
        record["content"] = "modified content"
        archive.write_text(json.dumps(record) + "\n")
        try:
            memoria.restore_raw_history(archive)
        except ValueError as exc:
            assert "checksum mismatch" in str(exc)
        else:
            raise AssertionError("modified archive must fail validation")
