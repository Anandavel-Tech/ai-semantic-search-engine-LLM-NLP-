"""Information Retrieval Evaluation Module.
Executes empirical benchmarks comparing Lexical, Semantic, and Hybrid retrieval modes
measuring Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Latency.
"""
import time
from typing import List, Dict, Any, Optional
from database import operations

# Academic benchmark query suite with relevance judgment keywords
BENCHMARK_QUERIES = [
    {
        "id": "Q1",
        "query": "How can AI detect diseases early in healthcare analytics?",
        "topic": "Healthcare AI",
        "target_keywords": ["disease", "healthcare", "diagnosis", "medical", "patient", "clinical"]
    },
    {
        "id": "Q2",
        "query": "Transformer attention mechanisms query key value self-attention",
        "topic": "Transformers & Attention",
        "target_keywords": ["transformer", "attention", "query", "key", "value", "softmax", "multi-head"]
    },
    {
        "id": "Q3",
        "query": "Inverted index posting lists boolean retrieval term frequency",
        "topic": "Information Retrieval",
        "target_keywords": ["inverted", "index", "posting", "vocabulary", "retrieval", "frequency"]
    },
    {
        "id": "Q4",
        "query": "Cybersecurity intrusion detection machine learning threat analysis",
        "topic": "Cybersecurity",
        "target_keywords": ["cybersecurity", "intrusion", "malware", "firewall", "threat", "anomaly"]
    },
    {
        "id": "Q5",
        "query": "Renewable energy forecasting solar wind climate science",
        "topic": "Climate & Energy",
        "target_keywords": ["renewable", "solar", "wind", "climate", "forecast", "photovoltaic"]
    },
    {
        "id": "Q6",
        "query": "Word2Vec skip-gram continuous bag of words embeddings",
        "topic": "Word Embeddings",
        "target_keywords": ["word2vec", "skip-gram", "cbow", "embedding", "gensim", "vector"]
    },
    {
        "id": "Q7",
        "query": "Deep learning convolutional neural networks computer vision",
        "topic": "Computer Vision",
        "target_keywords": ["convolutional", "cnn", "image", "vision", "filter", "pooling"]
    },
    {
        "id": "Q8",
        "query": "Natural language processing lemmatization tokenization preprocessing",
        "topic": "NLP Preprocessing",
        "target_keywords": ["tokenization", "lemmatization", "stemming", "stopword", "preprocessing", "nltk"]
    }
]

class EvaluationService:
    """Automated empirical benchmark evaluator for search engines."""
    
    def __init__(self, tfidf_service, embeddings_service, hybrid_service, inverted_index_service):
        self.tfidf = tfidf_service
        self.embeddings = embeddings_service
        self.hybrid = hybrid_service
        self.inverted_index = inverted_index_service

    def _is_relevant(self, chunk: Dict[str, Any], target_keywords: List[str]) -> bool:
        """Ground-truth relevance check based on topic keyword presence."""
        text = (chunk.get("text", "") + " " + chunk.get("doc_title", "")).lower()
        matches = sum(1 for kw in target_keywords if kw.lower() in text)
        return matches >= 2

    def evaluate_mode(self, mode: str, top_k: int = 5) -> Dict[str, Any]:
        """Run benchmark evaluation for a specific retrieval mode."""
        all_chunks = operations.get_all_chunks()
        if not all_chunks:
            return {
                "mode": mode,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "avg_latency_ms": 0.0,
                "query_count": 0,
                "per_query_results": []
            }
            
        p_at_k_list = []
        recall_at_k_list = []
        reciprocal_ranks = []
        latencies = []
        query_details = []
        
        for q_item in BENCHMARK_QUERIES:
            query = q_item["query"]
            targets = q_item["target_keywords"]
            
            # Ground-truth total relevant in corpus
            total_relevant_in_corpus = sum(
                1 for c in all_chunks 
                if self._is_relevant({"text": c["chunk_text"], "doc_title": c.get("doc_title", "")}, targets)
            )
            
            # Measure latency
            t0 = time.perf_counter()
            if mode == "keyword":
                results = self.tfidf.search(query, top_k=top_k)
            elif mode == "semantic":
                results = self.embeddings.search(query, top_k=top_k)
            elif mode == "inverted_index":
                results = self.inverted_index.search(query, operator="OR", top_k=top_k)
            else:  # hybrid
                results = self.hybrid.search(query, alpha=0.5, top_k=top_k)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)
            
            # Calculate metrics
            relevant_retrieved = 0
            first_relevant_rank = None
            
            for rank_idx, r in enumerate(results, start=1):
                if self._is_relevant(r, targets):
                    relevant_retrieved += 1
                    if first_relevant_rank is None:
                        first_relevant_rank = rank_idx
                        
            # Precision@K: relevant retrieved / K
            p_at_k = relevant_retrieved / max(top_k, 1)
            p_at_k_list.append(p_at_k)
            
            # Recall@K: relevant retrieved / total relevant
            rec_at_k = (relevant_retrieved / total_relevant_in_corpus) if total_relevant_in_corpus > 0 else 1.0
            recall_at_k_list.append(rec_at_k)
            
            # Reciprocal Rank
            rr = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
            reciprocal_ranks.append(rr)
            
            query_details.append({
                "query_id": q_item["id"],
                "query": query,
                "topic": q_item["topic"],
                "retrieved_count": len(results),
                "relevant_retrieved": relevant_retrieved,
                "precision_at_k": round(p_at_k, 3),
                "recall_at_k": round(rec_at_k, 3),
                "reciprocal_rank": round(rr, 3),
                "latency_ms": round(latency_ms, 2)
            })
            
        summary = {
            "mode": mode,
            "precision_at_k": round(float(sum(p_at_k_list) / max(len(p_at_k_list), 1)), 4),
            "recall_at_k": round(float(sum(recall_at_k_list) / max(len(recall_at_k_list), 1)), 4),
            "mrr": round(float(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)), 4),
            "avg_latency_ms": round(float(sum(latencies) / max(len(latencies), 1)), 2),
            "query_count": len(BENCHMARK_QUERIES),
            "per_query_results": query_details
        }
        
        # Save to database
        operations.save_evaluation_metrics(
            mode=mode,
            precision_at_k=summary["precision_at_k"],
            recall_at_k=summary["recall_at_k"],
            mrr=summary["mrr"],
            avg_latency_ms=summary["avg_latency_ms"],
            details=summary
        )
        
        return summary

    def run_full_benchmark(self, top_k: int = 5) -> Dict[str, Any]:
        """Execute benchmark comparison across all 4 retrieval modes."""
        modes = ["keyword", "semantic", "hybrid", "inverted_index"]
        results = {}
        for m in modes:
            results[m] = self.evaluate_mode(m, top_k=top_k)
            
        return {
            "top_k": top_k,
            "benchmark_query_count": len(BENCHMARK_QUERIES),
            "modes_evaluated": results
        }
