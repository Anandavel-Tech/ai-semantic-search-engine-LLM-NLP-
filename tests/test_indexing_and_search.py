"""Unit tests for Inverted Index, TF-IDF, Semantic Embeddings, and Hybrid Search."""
import pytest
from services.inverted_index import InvertedIndex
from services.tfidf_search import TfidfSearchService
from services.embeddings_service import EmbeddingsService
from services.hybrid_search import HybridSearchService

@pytest.fixture
def sample_chunks():
    return [
        {
            "id": 1,
            "document_id": 101,
            "doc_title": "Healthcare AI",
            "page_number": 1,
            "chunk_text": "Early disease detection using machine learning and medical data analytics."
        },
        {
            "id": 2,
            "document_id": 102,
            "doc_title": "Transformers NLP",
            "page_number": 1,
            "chunk_text": "Transformer architectures use multi-head self-attention mechanisms with query key value vectors."
        },
        {
            "id": 3,
            "document_id": 103,
            "doc_title": "Cybersecurity",
            "page_number": 1,
            "chunk_text": "Intrusion detection systems analyze network traffic anomalies and cyber attacks."
        }
    ]

def test_inverted_index_add_search_remove(sample_chunks):
    index = InvertedIndex()
    for c in sample_chunks:
        index.add_document(c["id"], c["chunk_text"], metadata=c)

    assert index.get_statistics()["indexed_chunks"] == 3
    
    # Search for disease
    hits = index.search("disease", operator="OR")
    assert len(hits) >= 1
    assert hits[0]["chunk_id"] == 1

    # Remove chunk 1
    index.remove_document(1)
    hits_after = index.search("disease", operator="OR")
    assert len(hits_after) == 0

def test_inverted_index_empty_query():
    index = InvertedIndex()
    assert index.search("") == []
    assert index.search("   ") == []

def test_tfidf_search(sample_chunks):
    tfidf = TfidfSearchService()
    tfidf.build_index(sample_chunks)
    assert tfidf.is_fitted

    results = tfidf.search("machine learning disease", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == 1
    assert results[0]["score"] > 0.0

def test_tfidf_search_out_of_vocab(sample_chunks):
    tfidf = TfidfSearchService()
    tfidf.build_index(sample_chunks)
    # Query with completely non-existent terms
    results = tfidf.search("xylophone pterodactyl", top_k=2)
    assert results == []

def test_semantic_search(sample_chunks):
    embeddings = EmbeddingsService()
    embeddings.build_index(sample_chunks)

    # Query with paraphrase: 'heart condition diagnosis' should match Healthcare AI even with different wording
    results = embeddings.search("diagnosing patient sickness", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == 1
    assert results[0]["score"] > 0.0

def test_hybrid_search(sample_chunks):
    tfidf = TfidfSearchService()
    tfidf.build_index(sample_chunks)

    embeddings = EmbeddingsService()
    embeddings.build_index(sample_chunks)

    hybrid = HybridSearchService(tfidf, embeddings)
    
    # Linear combination
    results_linear = hybrid.search("machine learning disease", alpha=0.6, top_k=2, fusion_strategy="linear")
    assert len(results_linear) >= 1
    assert "final_score" in results_linear[0]
    assert results_linear[0]["chunk_id"] == 1

    # RRF combination
    results_rrf = hybrid.search("transformer attention", top_k=2, fusion_strategy="rrf")
    assert len(results_rrf) >= 1
    assert results_rrf[0]["chunk_id"] == 2
