"""Unit tests for Database Operations and SQLite Schema."""
import uuid
import pytest
from database.db import init_db
from database import operations

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_insert_and_get_document():
    checksum = f"test_checksum_hash_{uuid.uuid4().hex}"
    doc_id = operations.insert_document(
        title="Test Ingestion Document",
        filename="test.txt",
        original_filename="test.txt",
        file_type="txt",
        file_size=1024,
        full_text="Sample text for database unit test.",
        word_count=6,
        checksum=checksum
    )
    assert doc_id > 0

    doc = operations.get_document_by_id(doc_id)
    assert doc is not None
    assert doc["title"] == "Test Ingestion Document"
    assert doc["checksum"] == checksum

    # Test checksum lookup
    existing = operations.get_document_by_checksum(checksum)
    assert existing is not None
    assert existing["id"] == doc_id

def test_cascade_delete_document():
    doc_id = operations.insert_document(
        title="Doc To Delete",
        filename="del.txt",
        original_filename="del.txt",
        file_type="txt",
        file_size=500,
        full_text="Text to delete.",
        word_count=3,
        checksum=f"delete_checksum_unique_{uuid.uuid4().hex}"
    )
    
    # Insert chunks
    chunks_data = [
        {"document_id": doc_id, "chunk_index": 0, "page_number": 1, "chunk_text": "Chunk A", "word_count": 2, "char_count": 7},
        {"document_id": doc_id, "chunk_index": 1, "page_number": 1, "chunk_text": "Chunk B", "word_count": 2, "char_count": 7}
    ]
    operations.insert_chunks(chunks_data)
    assert len(operations.get_chunks_for_document(doc_id)) == 2

    # Delete document
    deleted = operations.delete_document(doc_id)
    assert deleted is True
    assert operations.get_document_by_id(doc_id) is None
    # Chunks should be deleted via ON DELETE CASCADE
    assert len(operations.get_chunks_for_document(doc_id)) == 0

def test_search_history_logging():
    log_id = operations.log_search_query(
        query="early disease detection",
        search_mode="hybrid",
        top_k=5,
        alpha=0.5,
        result_count=4,
        response_time_ms=15.2
    )
    assert log_id > 0
    history = operations.get_search_history(limit=5)
    assert any(h["query"] == "early disease detection" for h in history)

def test_database_stats():
    stats = operations.get_database_stats()
    assert "total_documents" in stats
    assert "total_chunks" in stats
    assert "total_words" in stats
    assert "average_latency_ms" in stats
