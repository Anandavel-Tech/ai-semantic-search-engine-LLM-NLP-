"""Unit tests for Document Ingestion and Processing Pipeline."""
import pytest
import pymupdf
from services.document_processor import (
    compute_checksum,
    extract_text_from_pdf,
    extract_text_from_txt,
    chunk_text_sliding_window,
    process_file_bytes
)
from database.db import init_db

@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()

def test_compute_checksum():
    data1 = b"Hello World NLP"
    data2 = b"Hello World NLP"
    data3 = b"Different text"
    
    assert compute_checksum(data1) == compute_checksum(data2)
    assert compute_checksum(data1) != compute_checksum(data3)

def test_extract_text_from_txt():
    content = "Artificial intelligence transforms healthcare diagnostics with deep learning."
    raw = content.encode("utf-8")
    full_text, pages = extract_text_from_txt(raw)
    assert full_text == content
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1

def test_extract_text_from_pdf():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Sample medical diagnostics document for testing.")
    pdf_bytes = doc.tobytes()
    doc.close()

    full_text, pages = extract_text_from_pdf(pdf_bytes)
    assert "medical diagnostics" in full_text
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1

def test_chunk_text_sliding_window():
    words = [f"word{i}" for i in range(250)]
    page_text = " ".join(words)
    pages_data = [{"page_number": 1, "text": page_text}]

    chunks = chunk_text_sliding_window(pages_data, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[0]["page_number"] == 1
    # Check that overlap exists
    c0_words = set(chunks[0]["chunk_text"].split())
    c1_words = set(chunks[1]["chunk_text"].split())
    assert len(c0_words.intersection(c1_words)) > 0

def test_process_file_bytes_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        process_file_bytes(b"", "empty.txt")

def test_process_file_bytes_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        process_file_bytes(b"content", "file.docx")

def test_process_file_bytes_txt_success():
    import uuid
    content = f"Unique test document content {uuid.uuid4().hex} for automated testing verification.\n" * 10
    raw = content.encode("utf-8")
    res = process_file_bytes(raw, "unique_test_doc_1.txt", title="Test Doc 1")
    assert res["status"] == "success"
    assert res["chunk_count"] >= 1

    # Second ingestion of same bytes should return duplicate
    dup_res = process_file_bytes(raw, "unique_test_doc_1_copy.txt")
    assert dup_res["status"] == "duplicate"
