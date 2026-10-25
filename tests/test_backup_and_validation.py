"""Backup recovery, archive validation, and safe maintenance boundaries."""

import json
import sqlite3

import pytest

from memoria.core import Memoria


@pytest.fixture
def memory(tmp_path):
    with Memoria(tmp_path / 'memory.db', enable_embeddings=False) as value:
        yield value


def test_backup_includes_committed_wal_graph_raw_and_search(memory, tmp_path):
    stored = memory.remember('remember the zephyrmarker', metadata={'kind': 'preference'})
    destination = tmp_path / 'backups' / 'snapshot.db'
    result = memory.backup(destination)
    assert result['verified'] is True
    assert result['bytes'] > 0
    assert destination.stat().st_mode & 0o777 == 0o600
    assert not list(destination.parent.glob('.memoria-backup-*'))
    memory.remember('second memory')
    with Memoria(destination, enable_embeddings=False) as restored:
        assert restored.store.count() == 1
        assert restored.store.get(stored['conversation_id'])['metadata']['kind'] == 'preference'
        assert restored.recall('zephyrmarker').results
        assert restored.check_storage()['ok']


def test_backup_never_overwrites_existing_file_or_symlink(memory, tmp_path):
    existing = tmp_path / 'existing.db'
    existing.write_bytes(b'keep me')
    symlink = tmp_path / 'link.db'
    symlink.symlink_to(existing)
    for path in [existing, symlink, memory.db_path]:
        with pytest.raises(ValueError, match='exists'):
            memory.backup(path)
    assert existing.read_bytes() == b'keep me'


def test_backup_rejects_an_uncommitted_transaction(memory, tmp_path):
    memory.db.execute('BEGIN')
    with pytest.raises(ValueError, match='transaction'):
        memory.backup(tmp_path / 'snapshot.db')
    memory.db.rollback()


def test_integrity_check_detects_fts_drift(memory):
    memory.store.append('searchable provenance')
    memory.db.execute("INSERT INTO conversations_fts(conversations_fts) VALUES('delete-all')")
    memory.db.commit()
    result = memory.check_storage()
    assert result['integrity'] == ['ok']
    assert result['ok'] is False
    assert result['fts_error']


def test_backup_does_not_publish_a_damaged_snapshot(memory, tmp_path):
    memory.store.append('searchable provenance')
    memory.db.execute("INSERT INTO conversations_fts(conversations_fts) VALUES('delete-all')")
    memory.db.commit()
    destination = tmp_path / 'bad.db'
    with pytest.raises(ValueError, match='integrity'):
        memory.backup(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('.memoria-backup-*'))


@pytest.mark.parametrize('field', ['checksum_sha256', 'archive_format'])
def test_archive_cannot_bypass_verification_by_removing_fields(memory, tmp_path, field):
    memory.store.append('original')
    archive = tmp_path / 'archive.jsonl'
    memory.retain_raw_history(keep_latest=0, archive_path=archive, apply=True)
    record = json.loads(archive.read_text())
    record.pop(field)
    archive.write_text(json.dumps(record) + '\n')
    with pytest.raises(ValueError, match='checksum'):
        memory.restore_raw_history(archive, apply=True)
    assert memory.store.count() == 0


def test_retention_preserves_rows_if_archive_is_corrupted(memory, tmp_path):
    source = memory.store.append('keep this row')
    archive = tmp_path / 'corrupt.jsonl'
    archive.write_text('{truncated')
    with pytest.raises(ValueError, match='JSON'):
        memory.retain_raw_history(keep_latest=0, archive_path=archive, apply=True)
    assert memory.store.get(source) is not None
    assert memory.store.search_fts('keep')[0]['id'] == source


@pytest.mark.parametrize('value', ['false', 0, 1, None, []])
def test_retention_apply_requires_an_actual_boolean(memory, value):
    memory.store.append('keep this row')
    with pytest.raises(ValueError, match='boolean'):
        memory.retain_raw_history(keep_latest=0, apply=value)
    assert memory.store.count() == 1


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_retention_rejects_invalid_cutoffs(memory, value):
    with pytest.raises(ValueError):
        memory.retain_raw_history(older_than_days=value)


def test_utf8_usage_reports_bytes_not_characters(memory):
    content = 'café 日本語'
    memory.store.append(content)
    assert memory.storage_status()['database']['raw_content_bytes'] == len(content.encode())


def test_nonfinite_environment_quotas_do_not_crash(memory, monkeypatch):
    monkeypatch.setenv('MEMORIA_MAX_DB_MB', 'inf')
    monkeypatch.setenv('MEMORIA_HARD_MAX_DB_MB', 'nan')
    assert memory.storage_status()['database']['quota_bytes'] is None
    assert memory.remember('saved note')['triples_added'] == 1
