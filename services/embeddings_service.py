"""Sentence Embeddings and Semantic Search Module.
Uses Pretrained Sentence Transformers (all-MiniLM-L6-v2) to encode chunks and queries,
caches dense vectors to disk, and provides a graceful fallback projector if models are unavailable.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import Config
from services.preprocessor import preprocess_for_embeddings

class EmbeddingsService:
    """Production Sentence Transformer service with caching and graceful offline fallback."""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.model_loaded = False
        self.is_fallback = False
        self.fallback_vectorizer = None
        self.fallback_svd = None
        
        self.chunk_embeddings: Optional[np.ndarray] = None
        self.chunks: List[Dict[str, Any]] = []
        self.cache_file = Config.CACHE_DIR / "chunk_embeddings_cache.npz"

    def load_model(self) -> bool:
        """Attempt to load the pretrained Sentence Transformer model."""
        if self.model_loaded:
            return True
            
        try:
            from sentence_transformers import SentenceTransformer
            # Load model (uses local cache if already downloaded)
            self.model = SentenceTransformer(self.model_name)
            self.model_loaded = True
            self.is_fallback = False
            return True
        except Exception as e:
            # Fall back to dense TruncatedSVD semantic representation if offline/unavailable
            print(f"Warning: Could not load SentenceTransformer '{self.model_name}': {e}")
            print("Activating dense Latent Semantic Analysis (LSA) fallback projector.")
            self.model_loaded = False
            self.is_fallback = True
            return False

    def _encode_fallback(self, texts: List[str]) -> np.ndarray:
        """Deterministic dense semantic projection via L2-normalized LSA."""
        if not texts:
            return np.zeros((0, 128))
            
        if self.fallback_vectorizer is None:
            self.fallback_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf = self.fallback_vectorizer.fit_transform(texts)
            n_components = min(128, max(2, min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)))
            self.fallback_svd = TruncatedSVD(n_components=n_components, random_state=42)
            dense_vectors = self.fallback_svd.fit_transform(tfidf)
        else:
            tfidf = self.fallback_vectorizer.transform(texts)
            dense_vectors = self.fallback_svd.transform(tfidf)
            
        return normalize(dense_vectors, norm="l2")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into normalized dense embedding vectors."""
        clean_texts = [preprocess_for_embeddings(t) for t in texts]
        
        if not self.model_loaded and not self.is_fallback:
            self.load_model()
            
        if self.model_loaded and self.model:
            try:
                embeddings = self.model.encode(
                    clean_texts, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embeddings
            except Exception as e:
                print(f"Encoding failed with primary model: {e}. Switching to fallback.")
                self.is_fallback = True
                
        return self._encode_fallback(clean_texts)

    def build_index(self, chunks: List[Dict[str, Any]], force_recompute: bool = False) -> None:
        """Encode all document chunks and persist the embedding matrix to disk."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.chunk_embeddings = None
            return
            
        chunk_ids = np.array([c["id"] for c in self.chunks])
        
        # Check cache validity
        if not force_recompute and self.cache_file.exists():
            try:
                data = np.load(str(self.cache_file))
                cached_ids = data.get("chunk_ids")
                cached_emb = data.get("embeddings")
                if (cached_ids is not None and cached_emb is not None and 
                    len(cached_ids) == len(chunk_ids) and 
                    np.array_equal(cached_ids, chunk_ids)):
                    self.chunk_embeddings = cached_emb
                    return
            except Exception:
                pass  # Recompute if cache is corrupted
                
        texts = [c["chunk_text"] for c in self.chunks]
        self.chunk_embeddings = self.encode(texts)
        
        # Save cache
        try:
            Config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(self.cache_file),
                chunk_ids=chunk_ids,
                embeddings=self.chunk_embeddings
            )
        except Exception as e:
            print(f"Warning: Failed to persist embedding cache: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Rank chunks by cosine similarity between query embedding and chunk vectors."""
        if self.chunk_embeddings is None or not self.chunks:
            return []
            
        clean_query = preprocess_for_embeddings(query)
        if not clean_query.strip():
            return []
            
        query_vector = self.encode([clean_query])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.chunk_embeddings).flatten()
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            # Bound cosine similarity between 0 and 1
            calibrated_score = max(0.0, min(1.0, (score + 1.0) / 2.0 if self.is_fallback else score))
            
            if calibrated_score <= 0.001:
                break
                
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "doc_title": chunk.get("doc_title", "Document"),
                "original_filename": chunk.get("original_filename", ""),
                "page_number": chunk.get("page_number", 1),
                "text": chunk.get("chunk_text", ""),
                "score": round(calibrated_score, 4),
                "semantic_score": round(calibrated_score, 4),
                "model_used": "Fallback Dense LSA" if self.is_fallback else self.model_name
            })
            
            if len(results) >= top_k:
                break
                
        return results

    def group_by_document(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate chunk hits by parent document for document-level ranking."""
        doc_map = {}
        for r in results:
            doc_id = r["document_id"]
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "doc_title": r["doc_title"],
                    "original_filename": r["original_filename"],
                    "max_score": r["score"],
                    "matching_chunks": [r]
                }
            else:
                doc_map[doc_id]["matching_chunks"].append(r)
                if r["score"] > doc_map[doc_id]["max_score"]:
                    doc_map[doc_id]["max_score"] = r["score"]
                    
        grouped = list(doc_map.values())
        grouped.sort(key=lambda x: x["max_score"], reverse=True)
        return grouped
