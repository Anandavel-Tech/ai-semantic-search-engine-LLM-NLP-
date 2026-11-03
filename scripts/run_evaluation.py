"""Automated Benchmark Evaluation Runner.
Executes IR metrics across all 24 documents for Keyword (TF-IDF), Inverted Index,
Semantic (Sentence Transformers), and Hybrid Search.
Outputs a structured evaluation table and persists results to SQLite.
"""
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import operations
from services.tfidf_search import TfidfSearchService
from services.inverted_index import InvertedIndex
from services.embeddings_service import EmbeddingsService
from services.hybrid_search import HybridSearchService
from services.evaluation_service import EvaluationService

def run_evaluation_pipeline(top_k: int = 5):
    print("=" * 80)
    print("AI-POWERED SEMANTIC SEARCH ENGINE - EMPIRICAL BENCHMARK EVALUATION")
    print("=" * 80)
    
    # 1. Load all chunks from SQLite
    chunks = operations.get_all_chunks()
    if not chunks:
        print("Error: No document chunks found in database. Run create_sample_data.py first.")
        return
        
    print(f"Loaded {len(chunks)} chunks from database across {len(set(c['document_id'] for c in chunks))} documents.")
    
    # 2. Build Inverted Index
    print("\n[1/4] Building Inverted Index...")
    inv_index = InvertedIndex()
    for c in chunks:
        inv_index.add_document(
            c["id"], 
            c["chunk_text"], 
            metadata={
                "document_id": c["document_id"],
                "doc_title": c.get("doc_title", "Document"),
                "page_number": c.get("page_number", 1)
            }
        )
    print(f"Inverted Index ready: {len(inv_index.index)} unique vocabulary terms.")
    
    # 3. Fit TF-IDF Vectorizer
    print("\n[2/4] Fitting Scikit-Learn TF-IDF Vectorizer...")
    tfidf = TfidfSearchService()
    tfidf.build_index(chunks)
    print("TF-IDF Vector Space fitted successfully.")
    
    # 4. Build Dense Semantic Embeddings
    print("\n[3/4] Initializing Semantic Embeddings Service...")
    embeddings = EmbeddingsService()
    embeddings.build_index(chunks)
    print("Dense semantic embeddings indexed and cached.")
    
    # 5. Initialize Hybrid Retrieval
    hybrid = HybridSearchService(tfidf, embeddings)
    
    # 6. Execute Benchmark Suite
    print(f"\n[4/4] Executing Benchmark Queries (Top-K={top_k})...")
    evaluator = EvaluationService(
        tfidf_service=tfidf,
        embeddings_service=embeddings,
        hybrid_service=hybrid,
        inverted_index_service=inv_index
    )
    
    benchmark_report = evaluator.run_full_benchmark(top_k=top_k)
    modes_results = benchmark_report["modes_evaluated"]
    
    # Print formatted comparative table
    print("\n" + "=" * 80)
    print(f"{'RETRIEVAL MODE':<18} | {'PRECISION@' + str(top_k):<13} | {'RECALL@' + str(top_k):<12} | {'MRR':<10} | {'LATENCY (ms)':<14}")
    print("-" * 80)
    
    for mode_name, metrics in modes_results.items():
        mode_display = mode_name.upper().replace("_", " ")
        p_k = f"{metrics['precision_at_k']:.4f}"
        rec_k = f"{metrics['recall_at_k']:.4f}"
        mrr = f"{metrics['mrr']:.4f}"
        lat = f"{metrics['avg_latency_ms']:.2f} ms"
        print(f"{mode_display:<18} | {p_k:<13} | {rec_k:<12} | {mrr:<10} | {lat:<14}")
        
    print("=" * 80)
    
    # Save benchmark summary to JSON file in data/
    report_path = PROJECT_ROOT / "data" / "benchmark_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nDetailed evaluation report saved to: {report_path}")
    print("Results have also been committed to SQLite 'evaluation_metrics' table.")

if __name__ == "__main__":
    run_evaluation_pipeline(top_k=5)
