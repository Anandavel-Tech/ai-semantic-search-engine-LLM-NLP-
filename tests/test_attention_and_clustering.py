"""Unit tests for Attention Service, Word2Vec, and Clustering."""
import pytest
from services.attention_service import AttentionService
from services.word2vec_service import Word2VecService
from services.clustering_service import ClusteringService

def test_attention_service_extract():
    attn = AttentionService()
    res = attn.extract_attention(
        sentence="Artificial intelligence detects disease early in healthcare.",
        layer_idx=0,
        head_idx=0
    )
    assert "tokens" in res
    assert len(res["tokens"]) >= 5
    assert "attention_matrix" in res
    assert len(res["attention_matrix"]) == len(res["tokens"])
    assert len(res["attention_matrix"][0]) == len(res["tokens"])
    assert "heatmap_base64" in res
    assert len(res["heatmap_base64"]) > 100

def test_word2vec_service_train_and_similarity():
    w2v = Word2VecService(vector_size=30, min_count=1)
    texts = [
        "artificial intelligence and machine learning analyze healthcare records.",
        "doctors diagnose chronic disease using clinical healthcare analytics.",
        "neural networks and machine learning detect patterns in data."
    ]
    res = w2v.train(texts, epochs=10)
    assert res["status"] == "success"
    assert res["vocabulary_size"] > 5

    # Check vector retrieval
    vec = w2v.get_word_vector("machine", architecture="cbow")
    assert vec is not None
    assert len(vec) == 30

    # Check similarity
    sim = w2v.compute_similarity("machine", "learning", architecture="cbow")
    assert sim is not None
    assert -1.0 <= sim <= 1.0

def test_clustering_service_kmeans():
    docs = [
        {"id": 1, "title": "Doc 1", "full_text": "cardiovascular disease clinical patient diagnosis medicine hospital"},
        {"id": 2, "title": "Doc 2", "full_text": "kidney disease nephrology clinical patient healthcare diagnosis"},
        {"id": 3, "title": "Doc 3", "full_text": "transformer self attention query key value deep learning NLP"},
        {"id": 4, "title": "Doc 4", "full_text": "bert transformer language model attention encoder tokens"},
        {"id": 5, "title": "Doc 5", "full_text": "cybersecurity intrusion firewall network traffic packet attack malware"}
    ]
    clustering = ClusteringService()
    res = clustering.cluster_documents(docs, num_clusters=3, method="tfidf")
    assert res["status"] == "success"
    assert res["num_clusters"] == 3
    assert len(res["clusters"]) == 3
    assert len(res["plot_points"]) == 5
    assert -1.0 <= res["silhouette_score"] <= 1.0
