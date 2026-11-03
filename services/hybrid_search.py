"""Hybrid Information Retrieval and Ranking Module.
Combines Lexical (TF-IDF) and Dense Semantic (Sentence Transformers) retrieval
using Linear Score Combination and Reciprocal Rank Fusion (RRF).
"""
from typing import List, Dict, Any, Optional

class HybridSearchService:
    """Combines lexical keyword matching and dense semantic vector search."""
    
    def __init__(self, tfidf_service, embeddings_service):
        self.tfidf_service = tfidf_service
        self.embeddings_service = embeddings_service

    @staticmethod
    def _min_max_normalize(scores_dict: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores into the [0, 1] interval."""
        if not scores_dict:
            return {}
        vals = list(scores_dict.values())
        min_v, max_v = min(vals), max(vals)
        if max_v - min_v < 1e-6:
            return {k: 1.0 for k in scores_dict}
        return {k: (v - min_v) / (max_v - min_v) for k, v in scores_dict.items()}

    def search(
        self,
        query: str,
        alpha: float = 0.5,
        top_k: int = 5,
        fusion_strategy: str = "linear"  # 'linear' or 'rrf'
    ) -> List[Dict[str, Any]]:
        """Execute hybrid search combining TF-IDF and Semantic embeddings.
        
        Args:
            query: User's search query string.
            alpha: Weight for keyword score (0.0 to 1.0). Semantic weight is (1 - alpha).
            top_k: Maximum number of ranked results to return.
            fusion_strategy: 'linear' for weighted sum, 'rrf' for Reciprocal Rank Fusion.
        """
        # Clamp alpha between 0.0 and 1.0
        alpha = max(0.0, min(1.0, float(alpha)))
        candidate_pool_size = max(top_k * 3, 20)
        
        # 1. Retrieve candidates from both systems
        keyword_results = self.tfidf_service.search(query, top_k=candidate_pool_size)
        semantic_results = self.embeddings_service.search(query, top_k=candidate_pool_size)
        
        # Collect all unique candidate chunks
        all_chunks_map: Dict[int, Dict[str, Any]] = {}
        keyword_scores: Dict[int, float] = {}
        semantic_scores: Dict[int, float] = {}
        keyword_ranks: Dict[int, int] = {}
        semantic_ranks: Dict[int, int] = {}
        matched_terms_map: Dict[int, List[str]] = {}
        
        for rank, r in enumerate(keyword_results, start=1):
            cid = r["chunk_id"]
            all_chunks_map[cid] = r
            keyword_scores[cid] = r["score"]
            keyword_ranks[cid] = rank
            matched_terms_map[cid] = r.get("matched_terms", [])
            
        for rank, r in enumerate(semantic_results, start=1):
            cid = r["chunk_id"]
            if cid not in all_chunks_map:
                all_chunks_map[cid] = r
            semantic_scores[cid] = r["score"]
            semantic_ranks[cid] = rank
            
        if not all_chunks_map:
            return []
            
        combined_results = []
        
        if fusion_strategy.lower() == "rrf":
            # Reciprocal Rank Fusion: RRF_score = sum(1 / (k + rank))
            k_rrf = 60.0
            for cid, chunk in all_chunks_map.items():
                kw_rank = keyword_ranks.get(cid, 1000)
                sem_rank = semantic_ranks.get(cid, 1000)
                
                rrf_kw = 1.0 / (k_rrf + kw_rank) if kw_rank < 1000 else 0.0
                rrf_sem = 1.0 / (k_rrf + sem_rank) if sem_rank < 1000 else 0.0
                rrf_final = rrf_kw + rrf_sem
                
                combined_results.append({
                    "chunk_id": cid,
                    "document_id": chunk["document_id"],
                    "doc_title": chunk["doc_title"],
                    "original_filename": chunk.get("original_filename", ""),
                    "page_number": chunk.get("page_number", 1),
                    "text": chunk["text"],
                    "keyword_score": round(keyword_scores.get(cid, 0.0), 4),
                    "semantic_score": round(semantic_scores.get(cid, 0.0), 4),
                    "final_score": round(rrf_final, 5),
                    "matched_terms": matched_terms_map.get(cid, []),
                    "fusion_strategy": "Reciprocal Rank Fusion (RRF)"
                })
        else:
            # Linear combination: FinalScore = alpha * NormKeyword + (1 - alpha) * NormSemantic
            norm_kw = self._min_max_normalize(keyword_scores)
            norm_sem = self._min_max_normalize(semantic_scores)
            
            for cid, chunk in all_chunks_map.items():
                kw_val = norm_kw.get(cid, 0.0)
                sem_val = norm_sem.get(cid, 0.0)
                final_score = (alpha * kw_val) + ((1.0 - alpha) * sem_val)
                
                combined_results.append({
                    "chunk_id": cid,
                    "document_id": chunk["document_id"],
                    "doc_title": chunk["doc_title"],
                    "original_filename": chunk.get("original_filename", ""),
                    "page_number": chunk.get("page_number", 1),
                    "text": chunk["text"],
                    "keyword_score": round(keyword_scores.get(cid, 0.0), 4),
                    "semantic_score": round(semantic_scores.get(cid, 0.0), 4),
                    "normalized_keyword": round(kw_val, 4),
                    "normalized_semantic": round(sem_val, 4),
                    "final_score": round(final_score, 4),
                    "alpha": alpha,
                    "matched_terms": matched_terms_map.get(cid, []),
                    "fusion_strategy": f"Linear Combination (alpha={alpha})"
                })
                
        # Deterministic sort: descending score, ascending chunk_id for tie-breaking
        combined_results.sort(key=lambda x: (x["final_score"], -x["chunk_id"]), reverse=True)
        return combined_results[:top_k]
