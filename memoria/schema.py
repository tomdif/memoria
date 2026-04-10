"""Database schema and initialization."""

SCHEMA_SQL = """
-- Entities: nodes in the knowledge graph
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT,  -- person, project, concept, tool, file, etc.
    created_at REAL NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed REAL,
    embedding BLOB  -- numpy array serialized via tobytes()
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence);

-- Triples: directed edges with semantics
CREATE TABLE IF NOT EXISTS triples (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id TEXT REFERENCES entities(id),
    object_value TEXT,  -- for literal values (non-entity objects)
    relation_type TEXT NOT NULL DEFAULT 'fact'
        CHECK(relation_type IN ('fact', 'causal', 'supersedes', 'depends', 'contradicts')),
    confidence REAL NOT NULL DEFAULT 1.0,
    valid_from REAL,
    valid_until REAL,  -- NULL = still current
    source_ref TEXT,  -- conversation ID
    created_at REAL NOT NULL,
    last_accessed REAL,
    access_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject_id);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object_id);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_relation ON triples(relation_type);
CREATE INDEX IF NOT EXISTS idx_triples_confidence ON triples(confidence);
CREATE INDEX IF NOT EXISTS idx_triples_valid ON triples(valid_from, valid_until);

-- Raw conversation logs (Layer 1 — ground truth, never mutated)
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    role TEXT NOT NULL,  -- user, assistant, system
    content TEXT NOT NULL,
    session_id TEXT,
    metadata TEXT  -- JSON blob
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_time ON conversations(timestamp);

-- FTS5 virtual table for full-text search over conversations
CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
    content,
    content='conversations',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS conversations_ai AFTER INSERT ON conversations BEGIN
    INSERT INTO conversations_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Cluster summaries (Layer 3 — emergent from graph structure)
CREATE TABLE IF NOT EXISTS clusters (
    id TEXT PRIMARY KEY,
    entity_ids TEXT NOT NULL,  -- JSON array of entity IDs
    summary TEXT,
    centroid BLOB,  -- mean embedding
    created_at REAL NOT NULL,
    updated_at REAL
);

-- Consolidation log
CREATE TABLE IF NOT EXISTS consolidation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    action TEXT NOT NULL,  -- merge, decay, prune, cluster
    details TEXT  -- JSON
);
"""
