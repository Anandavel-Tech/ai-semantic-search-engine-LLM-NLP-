"""Document Ingestion and Processing Pipeline.
Handles validation, PDF/TXT text extraction via PyMuPDF, duplicate detection via SHA-256,
and overlapping sliding-window text chunking.
"""
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import pymupdf  # Modern PyMuPDF API

from config import Config
from database import operations

def compute_checksum(data: bytes) -> str:
    """Compute SHA-256 hex digest of file bytes for exact duplicate detection."""
    return hashlib.sha256(data).hexdigest()

def validate_uploaded_file(file: FileStorage) -> Tuple[bool, str, str]:
    """Validate uploaded file properties.
    
    Returns:
        (is_valid, error_message, file_type)
    """
    if not file or not file.filename:
        return False, "No file provided or filename is empty.", ""
        
    filename = secure_filename(file.filename)
    if not filename:
        return False, "Invalid filename provided.", ""
        
    ext = Path(filename).suffix.lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Only PDF and TXT are supported.", ""
        
    file_type = ext.lstrip(".")
    return True, "", file_type

def extract_text_from_pdf(data: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract text from PDF byte stream page-by-page using PyMuPDF.
    
    Returns:
        (full_text, pages_list)
    """
    pages_list = []
    full_text_parts = []
    
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Corrupted or invalid PDF file: {str(e)}")
        
    try:
        if len(doc) == 0:
            raise ValueError("The PDF document contains 0 pages.")
            
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()
            page_num = page_idx + 1
            if page_text:
                pages_list.append({"page_number": page_num, "text": page_text})
                full_text_parts.append(page_text)
                
        full_text = "\n\n".join(full_text_parts).strip()
        return full_text, pages_list
    finally:
        doc.close()

def extract_text_from_txt(data: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract text from TXT byte stream with robust encoding fallback."""
    # Attempt standard encodings
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = data.decode(encoding).strip()
            pages_list = [{"page_number": 1, "text": text}]
            return text, pages_list
        except UnicodeDecodeError:
            continue
            
    raise ValueError("Unable to decode text file. Unsupported character encoding.")

def chunk_text_sliding_window(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = Config.CHUNK_SIZE,
    chunk_overlap: int = Config.CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """Divide document text into overlapping sliding-window passages.
    
    Preserves:
    - chunk_index (sequential 0, 1, 2...)
    - page_number (mapped to origin page)
    - word_count and character count
    """
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)
        
    chunks = []
    global_chunk_idx = 0
    step = chunk_size - chunk_overlap
    
    for page in pages_data:
        page_num = page.get("page_number", 1)
        text = page.get("text", "").strip()
        if not text:
            continue
            
        words = text.split()
        if not words:
            continue
            
        # If page is shorter than chunk_size, emit single chunk
        if len(words) <= chunk_size:
            chunk_str = " ".join(words)
            chunks.append({
                "chunk_index": global_chunk_idx,
                "page_number": page_num,
                "chunk_text": chunk_str,
                "word_count": len(words),
                "char_count": len(chunk_str)
            })
            global_chunk_idx += 1
            continue
            
        # Sliding window across page words
        for start_idx in range(0, len(words), step):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            
            # Avoid trailing tiny fragments if overlap already covered them
            if len(chunk_words) < 10 and global_chunk_idx > 0:
                break
                
            chunk_str = " ".join(chunk_words)
            chunks.append({
                "chunk_index": global_chunk_idx,
                "page_number": page_num,
                "chunk_text": chunk_str,
                "word_count": len(chunk_words),
                "char_count": len(chunk_str)
            })
            global_chunk_idx += 1
            
            if end_idx == len(words):
                break
                
    return chunks

def process_file_bytes(
    raw_bytes: bytes,
    original_filename: str,
    title: Optional[str] = None
) -> Dict[str, Any]:
    """Ingest raw document bytes, extract text, chunk it, and persist to database.
    
    Returns:
        Dict with status, doc_id, chunk_count, and extracted preview.
    """
    file_size = len(raw_bytes)
    if file_size == 0:
        raise ValueError("Uploaded file is empty (0 bytes).")
        
    if file_size > Config.MAX_CONTENT_LENGTH:
        raise ValueError(
            f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit "
            f"({Config.MAX_CONTENT_LENGTH / (1024*1024):.1f} MB)."
        )
        
    ext = Path(original_filename).suffix.lower()
    file_type = ext.lstrip(".")
    if ext not in Config.ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'.")
        
    # Check duplicate via SHA-256
    checksum = compute_checksum(raw_bytes)
    existing = operations.get_document_by_checksum(checksum)
    if existing:
        return {
            "status": "duplicate",
            "message": f"Document already exists: '{existing['title']}' (ID: {existing['id']})",
            "document_id": existing["id"],
            "title": existing["title"]
        }
        
    # Extract text based on file format
    if file_type == "pdf":
        full_text, pages_data = extract_text_from_pdf(raw_bytes)
    else:
        full_text, pages_data = extract_text_from_txt(raw_bytes)
        
    if not full_text.strip():
        raise ValueError("Document contains no readable text (it may be an image scan or empty).")
        
    doc_title = title.strip() if title and title.strip() else Path(original_filename).stem.replace("_", " ").title()
    word_count = len(full_text.split())
    
    # Save file to disk
    stored_filename = f"{checksum[:12]}_{secure_filename(original_filename)}"
    save_path = Config.UPLOAD_DIR / stored_filename
    with open(save_path, "wb") as f:
        f.write(raw_bytes)
        
    # Insert document record in SQLite
    doc_id = operations.insert_document(
        title=doc_title,
        filename=stored_filename,
        original_filename=original_filename,
        file_type=file_type,
        file_size=file_size,
        full_text=full_text,
        word_count=word_count,
        checksum=checksum,
        status="processed"
    )
    
    # Generate sliding window chunks
    chunk_records = chunk_text_sliding_window(
        pages_data=pages_data,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    
    for c in chunk_records:
        c["document_id"] = doc_id
        
    # Persist chunks
    operations.insert_chunks(chunk_records)
    
    return {
        "status": "success",
        "document_id": doc_id,
        "title": doc_title,
        "word_count": word_count,
        "chunk_count": len(chunk_records),
        "preview": full_text[:400] + ("..." if len(full_text) > 400 else "")
    }
