"""Application configuration module.
Loads settings from environment variables or defines safe defaults.
"""
from pathlib import Path
import os
import certifi
from dotenv import load_dotenv

# Ensure Windows uses updated certifi CA bundle for secure SSL requests
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Base configuration class."""
    SECRET_KEY = os.getenv("SECRET_KEY", "btech_nlp_project_secret_key_2026_academic_demo")
    
    # Directory paths
    BASE_DIR = BASE_DIR
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = BASE_DIR / "uploads"
    CACHE_DIR = BASE_DIR / "cache"
    
    # Database
    DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "data/semantic_search.db")
    
    # Upload limits
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB
    ALLOWED_EXTENSIONS = {".pdf", ".txt"}
    
    # Chunking parameters (words)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 40))
    
    # Retrieval defaults
    DEFAULT_SEARCH_MODE = os.getenv("DEFAULT_SEARCH_MODE", "hybrid")
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
    DEFAULT_HYBRID_ALPHA = float(os.getenv("DEFAULT_HYBRID_ALPHA", 0.5))
    
    # Transformer & Embedding models
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    ATTENTION_MODEL_NAME = os.getenv("ATTENTION_MODEL_NAME", "bert-base-uncased")
    
    # LLM / RAG configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive_fallback")  # ollama, openai_compatible, extractive_fallback
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required runtime directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
