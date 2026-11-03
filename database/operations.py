"""Database CRUD Operations.
Provides parameterized, safe operations for documents, chunks, search logs, and metrics.
"""
from typing import List, Dict, Any, Optional
import json
from database.db import get_db_connection

def insert_document(
    title: str,
    filename: str,
    original_filename: str,
    file_type: str,
    file_size: int,
    full_text: str,
    word_count: int,
    checksum: str,
    status: str = "processed"
) -> int:
    """Insert a document record and return its new ID."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO documents (
                title, filename, original_filename, file_type, 
                file_size, full_text, word_count, checksum, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (title, filename, original_filename, file_type, 
             file_size, full_text, word_count, checksum, status)
        )
        doc_id = cur.lastrowid
        conn.commit()
        return doc_id
    finally:
        conn.close()

def get_document_by_checksum(checksum: str) -> Optional[Dict[str, Any]]:
    """Find document by SHA256 checksum for duplicate detection."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE checksum = ?", (checksum,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve full document details by ID."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_all_documents() -> List[Dict[str, Any]]:
    """Retrieve all documents with their chunk count."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT d.*, COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            GROUP BY d.id
            ORDER BY d.created_at DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

def delete_document(doc_id: int) -> bool:
    """Delete document by ID (cascades to chunks via foreign key)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()

def insert_chunks(chunks_data: List[Dict[str, Any]]) -> List[int]:
    """Bulk insert text chunks into database."""
    if not chunks_data:
        return []
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        inserted_ids = []
        for c in chunks_data:
            cur.execute(
                """
                INSERT INTO chunks (
                    document_id, chunk_index, page_number, 
                    chunk_text, word_count, char_count
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    c["document_id"],
                    c["chunk_index"],
                    c.get("page_number", 1),
                    c["chunk_text"],
                    c["word_count"],
                    c["char_count"]
                )
            )
            inserted_ids.append(cur.lastrowid)
        conn.commit()
        return inserted_ids
    finally:
        conn.close()

def get_chunks_for_document(doc_id: int) -> List[Dict[str, Any]]:
    """Retrieve all chunks belonging to a document ordered by index."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC", 
            (doc_id,)
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

def get_all_chunks() -> List[Dict[str, Any]]:
    """Retrieve all chunks along with parent document metadata."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.*, d.title AS doc_title, d.original_filename, d.file_type
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            ORDER BY c.document_id ASC, c.chunk_index ASC
            """
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

def get_chunk_by_id(chunk_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a single chunk with document metadata."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.*, d.title AS doc_title, d.original_filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id = ?
            """,
            (chunk_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def log_search_query(
    query: str,
    search_mode: str,
    top_k: int,
    alpha: float,
    result_count: int,
    response_time_ms: float
) -> int:
    """Record search query execution details."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO search_history (
                query, search_mode, top_k, alpha, result_count, response_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (query, search_mode, top_k, alpha, result_count, response_time_ms)
        )
        log_id = cur.lastrowid
        conn.commit()
        return log_id
    finally:
        conn.close()

def get_search_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent search queries."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM search_history 
            ORDER BY created_at DESC LIMIT ?
            """,
            (limit,)
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

def get_database_stats() -> Dict[str, Any]:
    """Calculate aggregate statistics across documents, chunks, and searches."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        
        # Document and chunk totals
        cur.execute("SELECT COUNT(*), COALESCE(SUM(word_count), 0) FROM documents")
        total_docs, total_words = cur.fetchone()
        
        cur.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cur.fetchone()[0]
        
        # Document types distribution
        cur.execute(
            "SELECT file_type, COUNT(*) FROM documents GROUP BY file_type"
        )
        type_dist = {row[0]: row[1] for row in cur.fetchall()}
        
        # Search counts and average latency
        cur.execute(
            "SELECT COUNT(*), COALESCE(AVG(response_time_ms), 0.0) FROM search_history"
        )
        total_searches, avg_latency = cur.fetchone()
        
        # Mode distribution
        cur.execute(
            "SELECT search_mode, COUNT(*) FROM search_history GROUP BY search_mode"
        )
        mode_dist = {row[0]: row[1] for row in cur.fetchall()}
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_words": total_words,
            "average_words_per_doc": round(total_words / total_docs, 1) if total_docs > 0 else 0,
            "file_type_distribution": type_dist,
            "total_searches": total_searches,
            "average_latency_ms": round(avg_latency, 2),
            "search_mode_distribution": mode_dist
        }
    finally:
        conn.close()

def save_evaluation_metrics(
    mode: str,
    precision_at_k: float,
    recall_at_k: float,
    mrr: float,
    avg_latency_ms: float,
    details: Optional[Dict[str, Any]] = None
) -> int:
    """Save an evaluation run for comparison."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO evaluation_metrics (
                mode, precision_at_k, recall_at_k, mrr, avg_latency_ms, details_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                mode, precision_at_k, recall_at_k, mrr, 
                avg_latency_ms, json.dumps(details or {})
            )
        )
        row_id = cur.lastrowid
        conn.commit()
        return row_id
    finally:
        conn.close()

def get_latest_evaluation_metrics() -> List[Dict[str, Any]]:
    """Retrieve latest evaluation runs for each mode."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM evaluation_metrics 
            ORDER BY evaluated_at DESC LIMIT 20
            """
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()
