"""Inverted Index and Document Representation Module.
Implements a functional inverted index mapping terms to posting lists with term frequencies
and positions. Also provides educational comparison across document representation paradigms.
"""
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple, Optional
import math
from services.preprocessor import normalize_text, tokenize_text, remove_stopwords, apply_lemmatization

class InvertedIndex:
    """In-memory Inverted Index with posting lists containing TF and positional offsets."""
    
    def __init__(self):
        # term -> {chunk_id: {"tf": int, "positions": List[int]}}
        self.index: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        # chunk_id -> {metadata: doc_title, text, word_count, doc_id}
        self.doc_map: Dict[int, Dict[str, Any]] = {}
        # Document frequencies: term -> count of docs containing term
        self.df: Dict[str, int] = defaultdict(int)
        
    def _tokenize(self, text: str) -> List[str]:
        """Normalize, tokenize, remove stopwords, and lemmatize for indexing."""
        norm = normalize_text(text, lowercase=True, remove_punctuation=True)
        tokens = tokenize_text(norm)
        filtered = remove_stopwords(tokens)
        return apply_lemmatization(filtered)

    def add_document(self, chunk_id: int, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Index a document chunk with positional posting lists."""
        # If already exists, remove it first
        if chunk_id in self.doc_map:
            self.remove_document(chunk_id)
            
        tokens = self._tokenize(text)
        self.doc_map[chunk_id] = {
            "text": text,
            "word_count": len(text.split()),
            "metadata": metadata or {}
        }
        
        seen_in_chunk: Set[str] = set()
        for pos, term in enumerate(tokens):
            if chunk_id not in self.index[term]:
                self.index[term][chunk_id] = {"tf": 0, "positions": []}
            self.index[term][chunk_id]["tf"] += 1
            self.index[term][chunk_id]["positions"].append(pos)
            
            if term not in seen_in_chunk:
                self.df[term] += 1
                seen_in_chunk.add(term)

    def remove_document(self, chunk_id: int) -> None:
        """Remove a document chunk from all posting lists and update document frequencies."""
        if chunk_id not in self.doc_map:
            return
            
        del self.doc_map[chunk_id]
        
        terms_to_prune = []
        for term, postings in self.index.items():
            if chunk_id in postings:
                del postings[chunk_id]
                self.df[term] = max(0, self.df[term] - 1)
                if not postings:
                    terms_to_prune.append(term)
                    
        for term in terms_to_prune:
            del self.index[term]
            if term in self.df:
                del self.df[term]

    def search(self, query: str, operator: str = "OR", top_k: int = 10) -> List[Dict[str, Any]]:
        """Search posting lists using Boolean or ranked TF-IDF scoring.
        
        Args:
            query: Query string.
            operator: 'AND' (must contain all terms) or 'OR' (match any term).
            top_k: Number of ranked hits to return.
        """
        query_terms = self._tokenize(query)
        if not query_terms or not self.doc_map:
            return []
            
        total_docs = len(self.doc_map)
        
        # Collect candidate chunks
        candidate_sets = []
        for term in query_terms:
            if term in self.index:
                candidate_sets.append(set(self.index[term].keys()))
            else:
                candidate_sets.append(set())
                
        if not candidate_sets:
            return []
            
        if operator.upper() == "AND":
            matching_chunk_ids = set.intersection(*candidate_sets) if candidate_sets else set()
        else:
            matching_chunk_ids = set.union(*candidate_sets) if candidate_sets else set()
            
        if not matching_chunk_ids:
            return []
            
        # Score candidates using TF-IDF weighting on inverted index
        scored_results = []
        for chunk_id in matching_chunk_ids:
            score = 0.0
            matched_terms = []
            chunk_info = self.doc_map[chunk_id]
            doc_len = max(chunk_info["word_count"], 1)
            
            for term in query_terms:
                if term in self.index and chunk_id in self.index[term]:
                    tf = self.index[term][chunk_id]["tf"]
                    # Log-normalized TF
                    norm_tf = (1 + math.log(tf)) / (1 + math.log(doc_len))
                    # Standard smoothed IDF
                    idf = math.log((1 + total_docs) / (1 + self.df[term])) + 1.0
                    score += norm_tf * idf
                    matched_terms.append(term)
                    
            meta = chunk_info.get("metadata", {})
            scored_results.append({
                "chunk_id": chunk_id,
                "document_id": meta.get("document_id"),
                "doc_title": meta.get("doc_title", f"Chunk #{chunk_id}"),
                "page_number": meta.get("page_number", 1),
                "text": chunk_info["text"],
                "score": round(score, 4),
                "matched_terms": matched_terms,
                "operator": operator.upper()
            })
            
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def get_posting_list(self, term: str) -> Dict[str, Any]:
        """Inspect the posting list for a specific term for educational inspection."""
        clean_term = apply_lemmatization(tokenize_text(normalize_text(term, lowercase=True)))
        query_term = clean_term[0] if clean_term else term.lower()
        
        postings = self.index.get(query_term, {})
        formatted_postings = []
        for chunk_id, data in postings.items():
            meta = self.doc_map.get(chunk_id, {}).get("metadata", {})
            formatted_postings.append({
                "chunk_id": chunk_id,
                "doc_title": meta.get("doc_title", f"Chunk {chunk_id}"),
                "term_frequency": data["tf"],
                "positions": data["positions"]
            })
            
        return {
            "term": query_term,
            "document_frequency": self.df.get(query_term, 0),
            "posting_count": len(postings),
            "postings": formatted_postings
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return structural statistics of the inverted index."""
        total_postings = sum(len(p) for p in self.index.values())
        vocab_size = len(self.index)
        return {
            "vocabulary_size": vocab_size,
            "indexed_chunks": len(self.doc_map),
            "total_posting_entries": total_postings,
            "average_postings_per_term": round(total_postings / max(vocab_size, 1), 2)
        }

    def compare_representations(self, sample_sentence: str) -> Dict[str, Any]:
        """Educational breakdown comparing 4 major text representation models.
        
        Compares:
        1. One-Hot Encoding
        2. Binary Term-Document Incidence
        3. TF-IDF Representation
        4. Dense Transformer Semantic Embeddings
        """
        words = sample_sentence.split()
        unique_words = sorted(list(set(w.lower() for w in words)))
        
        # 1. One-Hot Vector Matrix
        one_hot_vectors = {}
        vocab_len = len(unique_words)
        for idx, word in enumerate(unique_words):
            vec = [0] * vocab_len
            vec[idx] = 1
            one_hot_vectors[word] = vec
            
        # 2. Binary Incidence for chunks
        binary_incidence = {}
        for chunk_id, doc in list(self.doc_map.items())[:5]:
            doc_text = doc["text"].lower()
            binary_incidence[f"Chunk {chunk_id}"] = {
                word: (1 if word in doc_text else 0) for word in unique_words
            }
            
        return {
            "sample_sentence": sample_sentence,
            "vocabulary": unique_words,
            "vocabulary_size": vocab_len,
            "one_hot_encoding": {
                "description": "Orthogonal vectors where only one dimension is 1 and all others are 0.",
                "disadvantage": "Sparse, high-dimensional, orthogonal (dot product is 0, zero semantic similarity).",
                "vectors": one_hot_vectors
            },
            "binary_incidence": {
                "description": "Matrix tracking presence (1) or absence (0) of terms across documents.",
                "matrix": binary_incidence
            },
            "tfidf_explanation": {
                "description": "Weights terms proportionally to within-doc frequency, inversely to collection frequency.",
                "formula": "TF-IDF(t, d) = TF(t, d) * log((1 + N) / (1 + DF(t)))",
                "advantage": "Downweights universally common words like 'the', highlights discriminative domain terms."
            },
            "dense_embeddings_explanation": {
                "description": "Fixed-dimension continuous vectors (e.g. 384 dimensions) learned via neural attention.",
                "advantage": "Captures conceptual synonyms, context-dependent semantics, and semantic distance via cosine similarity."
            }
        }
