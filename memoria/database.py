"""Composable SQLite transactions for standalone and multi-step writes."""

from contextlib import contextmanager
from functools import wraps
import sqlite3
import uuid


@contextmanager
def transaction(db: sqlite3.Connection):
    """Commit a complete operation, or roll it back without committing its caller.

    Reserve the writer before read/modify/write operations so separate processes
    cannot race entity lookup or contradiction detection. Nested operations use
    savepoints and never commit an enclosing transaction.
    """
    nested = db.in_transaction
    savepoint = "memoria_" + uuid.uuid4().hex
    if nested:
        db.execute(f"SAVEPOINT {savepoint}")
    else:
        db.execute("BEGIN IMMEDIATE")
    try:
        yield
        if nested:
            db.execute(f"RELEASE SAVEPOINT {savepoint}")
        else:
            db.commit()
    except BaseException:
        if nested:
            db.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            db.execute(f"RELEASE SAVEPOINT {savepoint}")
        else:
            db.rollback()
        raise


def transactional(method):
    """Wrap a method on an object with a ``db`` connection in a transaction."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with transaction(self.db):
            return method(self, *args, **kwargs)
    return wrapped
