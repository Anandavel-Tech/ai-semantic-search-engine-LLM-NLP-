"""TF-IDF Information Retrieval Module using Scikit-Learn.
Builds and maintains the document-term matrix, executes cosine similarity ranking,
and gracefully handles empty collections, out-of-vocabulary queries, and incremental index updates.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from services.preprocessor import preprocess_for_tfidf

class TfidfSearchService:
    """Production-grade TF-IDF Search Engine using Scikit-learn."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            max_features=15000,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.tfidf_matrix = None
        self.chunks: List[Dict[str, Any]] = []
        self.is_fitted = False

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Fit the vectorizer on all chunk texts and store the sparse TF-IDF matrix."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.tfidf_matrix = None
            self.is_fitted = False
            return
            
        corpus = [preprocess_for_tfidf(c["chunk_text"]) for c in self.chunks]
        
        # Guard against corpus with only empty tokens
        if not any(text.strip() for text in corpus):
            self.tfidf_matrix = None
            self.is_fitted = False
            return
            
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Rank document chunks by cosine similarity to the query vector.
        
        Handles:
        - Empty collection
        - Empty or whitespace query
        - Out-of-vocabulary query terms
        """
        if not self.is_fitted or self.tfidf_matrix is None or not self.chunks:
            return []
            
        clean_query = preprocess_for_tfidf(query)
        if not clean_query.strip():
            return []
            
        # Transform query into TF-IDF vector
        try:
            query_vector = self.vectorizer.transform([clean_query])
        except Exception:
            return []
            
        # Check if query vector is all zeros (completely out of vocabulary)
        if query_vector.nnz == 0:
            return []
            
        # Compute cosine similarity
        cosine_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Extract vocabulary terms for query explanation
        feature_names = self.vectorizer.get_feature_names_out()
        query_feature_indices = query_vector.indices
        query_terms = [feature_names[i] for i in query_feature_indices]
        
        # Rank descending
        top_indices = np.argsort(cosine_scores)[::-1]
        
        results = []
        for idx in top_indices:
            score = float(cosine_scores[idx])
            if score <= 0.0001:
                break
                
            chunk = self.chunks[idx]
            
            # Identify matched keywords in this specific chunk
            chunk_vector = self.tfidf_matrix[idx]
            matched = [
                feature_names[i] for i in query_feature_indices 
                if chunk_vector[0, i] > 0
            ]
            
            results.append({
                "chunk_id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "doc_title": chunk.get("doc_title", "Document"),
                "original_filename": chunk.get("original_filename", ""),
                "page_number": chunk.get("page_number", 1),
                "text": chunk.get("chunk_text", ""),
                "score": round(score, 4),
                "keyword_score": round(score, 4),
                "matched_terms": matched
            })
            
            if len(results) >= top_k:
                break
                
        return results

    def get_top_terms(self, top_n: int = 25) -> List[Dict[str, Any]]:
        """Return highest IDF terms across corpus for vocabulary inspection."""
        if not self.is_fitted:
            return []
        feature_names = self.vectorizer.get_feature_names_out()
        idfs = self.vectorizer.idf_
        indices = np.argsort(idfs)[::-1]  # Highest IDF first
        
        return [
            {"term": feature_names[i], "idf": round(float(idfs[i]), 3)}
            for i in indices[:top_n]
        ]
