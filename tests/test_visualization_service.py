"""Unit tests for Visualization Service (Word Cloud, Bigram Network Graph, Bar Chart)."""
import pytest
from services.visualization_service import VisualizationService
from app import create_app

@pytest.fixture
def viz_service():
    return VisualizationService()

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_extract_cleaned_tokens(viz_service):
    text = "Machine learning and artificial intelligence detect diseases early in clinical healthcare."
    tokens = viz_service.extract_cleaned_tokens(text)
    assert len(tokens) >= 5
    assert "machine" in tokens
    assert "learning" in tokens
    assert "and" not in tokens  # Stopword removed

def test_extract_bigrams(viz_service):
    tokens = ["machine", "learning", "deep", "learning", "neural", "network"]
    bigrams = viz_service.extract_bigrams(tokens)
    assert len(bigrams) == 5
    assert ("machine", "learning") in bigrams
    assert ("neural", "network") in bigrams

def test_generate_word_cloud(viz_service):
    text = "Artificial intelligence machine learning deep learning neural network data analytics."
    base64_img, top_words = viz_service.generate_word_cloud(text, max_words=20)
    assert isinstance(base64_img, str)
    assert len(base64_img) > 100
    assert len(top_words) >= 4
    assert any(item["word"] == "learning" for item in top_words)

def test_generate_bigram_network(viz_service):
    text = (
        "Machine learning models predict disease. Deep learning models detect disease. "
        "Machine learning optimizes neural network structures."
    )
    base64_img, top_bigrams = viz_service.generate_bigram_network(text, top_n=10)
    assert isinstance(base64_img, str)
    assert len(base64_img) > 100
    assert len(top_bigrams) >= 1
    assert any("learning" in b["bigram"] for b in top_bigrams)

def test_generate_barchart(viz_service):
    text = "Machine learning deep learning reinforcement learning neural network."
    # Test unigram
    base64_unigram, data_unigram = viz_service.generate_barchart(text, chart_type="unigram", top_k=5)
    assert isinstance(base64_unigram, str)
    assert len(base64_unigram) > 100
    assert len(data_unigram) >= 3

    # Test bigram
    base64_bigram, data_bigram = viz_service.generate_barchart(text, chart_type="bigram", top_k=5)
    assert isinstance(base64_bigram, str)
    assert len(base64_bigram) > 100
    assert len(data_bigram) >= 2

def test_visualizations_flask_route(client):
    # Test corpus-wide visualization
    res = client.get("/visualizations")
    assert res.status_code == 200
    assert b"NLP Text Visualizations" in res.data
    assert b"Word Cloud" in res.data
    assert b"Bigram Co-occurrence" in res.data
    assert b"Frequency Bar Chart" in res.data

    # Test document-specific visualization
    res_doc = client.get("/visualizations?doc_id=1&bar_type=bigram&top_n=10")
    assert res_doc.status_code == 200
    assert b"Word Cloud" in res_doc.data
