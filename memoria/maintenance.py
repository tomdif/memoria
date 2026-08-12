"""Capacity reporting and explicit, archive-backed storage maintenance."""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def database_usage(db: sqlite3.Connection, db_path: str | Path) -> dict[str, Any]:
    """Return logical and physical usage for one open Memoria database."""
    path = Path(db_path).expanduser()
    page_size = int(db.execute("PRAGMA page_size").fetchone()[0])
    page_count = int(db.execute("PRAGMA page_count").fetchone()[0])
    free_pages = int(db.execute("PRAGMA freelist_count").fetchone()[0])
    wal_path = Path(f"{path}-wal")
    shm_path = Path(f"{path}-shm")
    db_bytes = _file_size(path)
    wal_bytes = _file_size(wal_path)
    shm_bytes = _file_size(shm_path)
    content_bytes = int(db.execute(
        "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM conversations"
    ).fetchone()[0])
    embedding_bytes = int(db.execute(
        "SELECT COALESCE(SUM(LENGTH(embedding)), 0) FROM entities"
    ).fetchone()[0])
    conversations = int(db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
    entities = int(db.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
    triples = int(db.execute("SELECT COUNT(*) FROM triples").fetchone()[0])
    quota_mb = _positive_float(os.environ.get("MEMORIA_MAX_DB_MB"))
    quota_bytes = int(quota_mb * 1024 * 1024) if quota_mb is not None else None
    hard_quota_mb = _positive_float(os.environ.get("MEMORIA_HARD_MAX_DB_MB"))
    hard_quota_bytes = (
        int(hard_quota_mb * 1024 * 1024) if hard_quota_mb is not None else None
    )
    physical_bytes = db_bytes + wal_bytes + shm_bytes
    return {
        "db_path": str(path),
        "db_bytes": db_bytes,
        "wal_bytes": wal_bytes,
        "shm_bytes": shm_bytes,
        "physical_bytes": physical_bytes,
        "logical_bytes": (page_count - free_pages) * page_size,
        "reclaimable_bytes": free_pages * page_size,
        "raw_content_bytes": content_bytes,
        "embedding_bytes": embedding_bytes,
        "conversations": conversations,
        "entities": entities,
        "triples": triples,
        "quota_bytes": quota_bytes,
        "quota_exceeded": quota_bytes is not None and physical_bytes > quota_bytes,
        "quota_percent": (
            round(physical_bytes / quota_bytes * 100, 2)
            if quota_bytes
            else None
        ),
        "hard_quota_bytes": hard_quota_bytes,
        "hard_quota_exceeded": (
            hard_quota_bytes is not None and physical_bytes >= hard_quota_bytes
        ),
    }


def _positive_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def hard_quota_bytes() -> int | None:
    value = _positive_float(os.environ.get("MEMORIA_HARD_MAX_DB_MB"))
    return int(value * 1024 * 1024) if value is not None else None


def all_database_usage(
    base_db_path: str | Path,
    *,
    current_path: str | Path | None = None,
    current_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate the global database, project scopes, and default archives."""
    base = Path(base_db_path).expanduser().resolve()
    paths = []
    if base.is_file() and not base.is_symlink():
        paths.append(base)
    scopes_dir = base.parent / "scopes"
    if scopes_dir.is_dir() and not scopes_dir.is_symlink():
        paths.extend(
            path.resolve()
            for path in scopes_dir.glob("*.db")
            if path.is_file() and not path.is_symlink()
        )

    resolved_current = (
        Path(current_path).expanduser().resolve() if current_path is not None else None
    )
    databases = []
    errors = []
    for path in sorted(set(paths), key=str):
        if resolved_current == path and current_usage is not None:
            databases.append(current_usage)
            continue
        try:
            connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
            try:
                databases.append(database_usage(connection, path))
            finally:
                connection.close()
        except (OSError, sqlite3.Error) as exc:
            errors.append({"db_path": str(path), "error": str(exc)})

    archive_files = []
    for archive_dir in (base.parent / "archives", scopes_dir / "archives"):
        if archive_dir.is_dir() and not archive_dir.is_symlink():
            archive_files.extend(
                path for path in archive_dir.glob("*.jsonl")
                if path.is_file() and not path.is_symlink()
            )
    archive_bytes = sum(_file_size(path) for path in archive_files)
    return {
        "databases": databases,
        "database_count": len(databases),
        "database_bytes": sum(item["physical_bytes"] for item in databases),
        "default_archive_files": len(archive_files),
        "default_archive_bytes": archive_bytes,
        "total_managed_bytes": (
            sum(item["physical_bytes"] for item in databases) + archive_bytes
        ),
        "errors": errors,
    }


def pending_state_usage(state_dir: str | Path) -> dict[str, int | str]:
    path = Path(state_dir).expanduser()
    files = list(path.glob("*.json")) if path.exists() else []
    return {
        "path": str(path),
        "files": len(files),
        "bytes": sum(_file_size(item) for item in files),
    }


def clean_stale_pending(
    state_dir: str | Path,
    *,
    older_than_days: float = 7,
    dry_run: bool = True,
) -> dict[str, Any]:
    if older_than_days < 0:
        raise ValueError("older_than_days must be non-negative")
    path = Path(state_dir).expanduser()
    cutoff = time.time() - older_than_days * 86400
    candidates = []
    if path.exists():
        for item in path.glob("*.json"):
            try:
                if item.stat().st_mtime < cutoff and item.is_file() and not item.is_symlink():
                    candidates.append(item)
            except OSError:
                continue
    bytes_found = sum(_file_size(item) for item in candidates)
    if not dry_run:
        for item in candidates:
            item.unlink()
    return {
        "dry_run": dry_run,
        "files": len(candidates),
        "bytes": bytes_found,
        "state_dir": str(path),
    }


def default_archive_path(db_path: str | Path) -> Path:
    path = Path(db_path).expanduser()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return path.parent / "archives" / f"{path.stem}-{stamp}.jsonl"


def compact_database(
    db: sqlite3.Connection,
    db_path: str | Path,
    *,
    vacuum: bool = False,
) -> dict[str, Any]:
    """Checkpoint WAL and optionally VACUUM after verifying temporary capacity."""
    path = Path(db_path).expanduser()
    before = database_usage(db, path)
    checkpoint = list(db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
    vacuumed = False
    if vacuum:
        required = max(_file_size(path), 1)
        available = shutil.disk_usage(path.parent).free
        if available < required:
            raise OSError(
                f"VACUUM needs about {required} free bytes; only {available} available"
            )
        db.execute("VACUUM")
        vacuumed = True
    after = database_usage(db, path)
    return {
        "checkpoint": checkpoint,
        "vacuumed": vacuumed,
        "before_bytes": before["physical_bytes"],
        "after_bytes": after["physical_bytes"],
        "reclaimed_bytes": max(before["physical_bytes"] - after["physical_bytes"], 0),
    }
