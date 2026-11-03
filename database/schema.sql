-- Database Schema for Semantic Search Engine
PRAGMA foreign_keys = ON;

-- Documents Table
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_type TEXT NOT NULL,       -- 'pdf' or 'txt'
    file_size INTEGER NOT NULL,     -- bytes
    full_text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    checksum TEXT UNIQUE NOT NULL, -- SHA256 for duplicate detection
    status TEXT DEFAULT 'processed',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document Chunks Table (for passage-level retrieval and RAG context)
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER DEFAULT 1,
    chunk_text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- Search History Logging
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    search_mode TEXT NOT NULL,       -- 'keyword', 'semantic', 'hybrid', 'inverted_index'
    top_k INTEGER NOT NULL,
    alpha REAL DEFAULT 0.5,
    result_count INTEGER NOT NULL,
    response_time_ms REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Evaluation Metrics Archive
CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL,
    precision_at_k REAL NOT NULL,
    recall_at_k REAL NOT NULL,
    mrr REAL NOT NULL,
    avg_latency_ms REAL NOT NULL,
    details_json TEXT,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_docs_checksum ON documents(checksum);
CREATE INDEX IF NOT EXISTS idx_history_created ON search_history(created_at);
