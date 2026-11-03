"""End-to-End integration tests for all Flask routes and web endpoints."""
import io
import pytest
from app import create_app
from database.db import init_db

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Intelligent Document Search" in response.data

def test_search_route(client):
    response = client.get("/search?query=disease&mode=keyword&top_k=3")
    assert response.status_code == 200
    assert b"Results for:" in response.data

def test_qa_route(client):
    response = client.get("/qa?question=How+can+AI+detect+diseases+early?")
    assert response.status_code == 200
    assert b"Synthesized Answer" in response.data

def test_upload_get_page(client):
    response = client.get("/upload")
    assert response.status_code == 200
    assert b"Upload Documents to Corpus" in response.data

def test_upload_post_valid_file(client):
    import uuid
    uid = uuid.uuid4().hex[:6]
    data = {
        "files": (io.BytesIO(f"Synthetic testing text {uid} for automated route upload verification.".encode("utf-8")), f"route_test_{uid}.txt"),
        "title": f"Route Test Document {uid}"
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Indexed Documents" in response.data or f"Route Test Document {uid}".encode("utf-8") in response.data

def test_documents_list_route(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert b"Document Corpus Management" in response.data

def test_attention_route(client):
    response = client.get("/attention")
    assert response.status_code == 200
    assert b"Attention Weight Heatmap" in response.data

def test_word2vec_route(client):
    response = client.get("/word2vec")
    assert response.status_code == 200
    assert b"Word2Vec Exploration Laboratory" in response.data

def test_clustering_route(client):
    response = client.get("/clustering?k=3&method=tfidf")
    assert response.status_code == 200
    assert b"2D PCA Cluster Projection" in response.data

def test_analytics_route(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    assert b"Search Analytics" in response.data

def test_settings_route(client):
    response = client.get("/settings")
    assert response.status_code == 200
    assert b"System Configuration" in response.data

def test_health_check_route(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True
    assert data["document_count"] >= 1
