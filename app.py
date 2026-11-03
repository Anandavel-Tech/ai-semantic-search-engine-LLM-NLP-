"""AI-Powered Semantic Search Engine Using LLMs and Attention Mechanisms.
Main Flask Web Application and REST API Gateway.
"""
import time
from pathlib import Path
from typing import Dict, Any, List
from flask import (
    Flask, render_template, request, redirect, 
    url_for, flash, jsonify, abort
)

from config import Config
from database.db import init_db
from database import operations
from services.document_processor import process_file_bytes
from services.inverted_index import InvertedIndex
from services.tfidf_search import TfidfSearchService
from services.embeddings_service import EmbeddingsService
from services.hybrid_search import HybridSearchService
from services.word2vec_service import Word2VecService
from services.attention_service import AttentionService
from services.clustering_service import ClusteringService
from services.llm_service import LLMService
from services.summarization_service import SummarizationService
from services.evaluation_service import EvaluationService
from services.visualization_service import VisualizationService

def create_app() -> Flask:
    """Initialize and configure the Flask application."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = Config.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
    
    # Ensure database and filesystem directories exist
    Config.ensure_directories()
    init_db()
    
    # -------------------------------------------------------------
    # Singleton In-Memory NLP & Retrieval Engines
    # -------------------------------------------------------------
    inv_index = InvertedIndex()
    tfidf_service = TfidfSearchService()
    embeddings_service = EmbeddingsService()
    hybrid_service = HybridSearchService(tfidf_service, embeddings_service)
    word2vec_service = Word2VecService()
    attention_service = AttentionService()
    clustering_service = ClusteringService(tfidf_service, embeddings_service)
    llm_service = LLMService()
    summarizer_service = SummarizationService(llm_service)
    evaluator_service = EvaluationService(tfidf_service, embeddings_service, hybrid_service, inv_index)
    visualization_service = VisualizationService()

    def rebuild_all_indexes():
        """Re-synchronize in-memory search indexes with persisted database chunks."""
        chunks = operations.get_all_chunks()
        if not chunks:
            return
            
        # 1. Rebuild Inverted Index
        for c in chunks:
            inv_index.add_document(
                c["id"],
                c["chunk_text"],
                metadata={
                    "document_id": c["document_id"],
                    "doc_title": c.get("doc_title", "Document"),
                    "original_filename": c.get("original_filename", ""),
                    "page_number": c.get("page_number", 1)
                }
            )
            
        # 2. Build Scikit-Learn TF-IDF Index
        tfidf_service.build_index(chunks)
        
        # 3. Build Dense Semantic Embeddings Index
        embeddings_service.build_index(chunks)
        
        # 4. Train Gensim Word2Vec on corpus
        corpus_texts = [c["chunk_text"] for c in chunks]
        word2vec_service.train(corpus_texts)

    # Initial index synchronization
    rebuild_all_indexes()

    # -------------------------------------------------------------
    # Web Routes & Controller Logic
    # -------------------------------------------------------------
    
    @app.route("/")
    def home():
        """Homepage and search portal."""
        stats = operations.get_database_stats()
        return render_template(
            "index.html", 
            active_page="home",
            stats=stats,
            mode=Config.DEFAULT_SEARCH_MODE,
            top_k=Config.DEFAULT_TOP_K,
            alpha=Config.DEFAULT_HYBRID_ALPHA
        )

    @app.route("/search")
    def search_action():
        """Execute query across configured search engines."""
        query = request.args.get("query", "").strip()
        mode = request.args.get("mode", Config.DEFAULT_SEARCH_MODE).lower()
        top_k = int(request.args.get("top_k", Config.DEFAULT_TOP_K))
        alpha = float(request.args.get("alpha", Config.DEFAULT_HYBRID_ALPHA))
        fusion = request.args.get("fusion", "linear")

        if not query:
            flash("Please enter a search query.", "warning")
            return redirect(url_for("home"))

        t0 = time.perf_counter()
        results: List[Dict[str, Any]] = []

        if mode == "semantic":
            results = embeddings_service.search(query, top_k=top_k)
        elif mode == "keyword":
            results = tfidf_service.search(query, top_k=top_k)
        elif mode == "inverted_index":
            results = inv_index.search(query, operator="OR", top_k=top_k)
        else:  # hybrid
            results = hybrid_service.search(query, alpha=alpha, top_k=top_k, fusion_strategy=fusion)

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 2)

        # Log query to database
        operations.log_search_query(
            query=query,
            search_mode=mode,
            top_k=top_k,
            alpha=alpha,
            result_count=len(results),
            response_time_ms=latency_ms
        )

        stats = operations.get_database_stats()
        return render_template(
            "index.html",
            active_page="home",
            query=query,
            mode=mode,
            top_k=top_k,
            alpha=alpha,
            fusion=fusion,
            results=results,
            latency_ms=latency_ms,
            stats=stats
        )

    @app.route("/qa")
    def qa_page():
        """RAG Question Answering interface."""
        question = request.args.get("question", "").strip()
        if not question:
            return render_template("qa.html", active_page="qa")
        return qa_action()

    @app.route("/qa/ask")
    def qa_action():
        """RAG pipeline: Retrieve relevant passages, assemble context, generate answer."""
        question = request.args.get("question", "").strip()
        if not question:
            flash("Please enter a question.", "warning")
            return redirect(url_for("qa_page"))

        # Retrieve top passages using hybrid search
        retrieved_passages = hybrid_service.search(question, alpha=0.5, top_k=4)
        
        # Generate answer with citations
        answer_data = llm_service.generate_rag_answer(question, retrieved_passages)

        return render_template(
            "qa.html",
            active_page="qa",
            question=question,
            answer_data=answer_data,
            retrieved_chunks=retrieved_passages
        )

    @app.route("/upload")
    def upload_page():
        """Document upload page."""
        return render_template("upload.html", active_page="upload")

    @app.route("/upload", methods=["POST"])
    def upload_action():
        """Process multiple file uploads with validation, extraction, chunking, and indexing."""
        uploaded_files = request.files.getlist("files")
        single_title = request.form.get("title", "").strip()

        if not uploaded_files or (len(uploaded_files) == 1 and uploaded_files[0].filename == ""):
            flash("No files were selected for upload.", "danger")
            return redirect(url_for("upload_page"))

        success_count = 0
        duplicate_count = 0
        error_messages = []

        for file_storage in uploaded_files:
            if not file_storage or not file_storage.filename:
                continue

            raw_bytes = file_storage.read()
            title = single_title if (len(uploaded_files) == 1 and single_title) else None

            try:
                res = process_file_bytes(
                    raw_bytes=raw_bytes,
                    original_filename=file_storage.filename,
                    title=title
                )
                if res.get("status") == "duplicate":
                    duplicate_count += 1
                else:
                    success_count += 1
            except Exception as e:
                error_messages.append(f"{file_storage.filename}: {str(e)}")

        # Re-sync search indexes with newly added documents
        if success_count > 0:
            rebuild_all_indexes()

        if success_count > 0:
            flash(f"Successfully processed and indexed {success_count} document(s).", "success")
        if duplicate_count > 0:
            flash(f"{duplicate_count} file(s) skipped as exact duplicates (SHA-256 matched).", "info")
        for err in error_messages:
            flash(f"Upload error: {err}", "danger")

        return redirect(url_for("documents_page"))

    @app.route("/documents")
    def documents_page():
        """List and manage all documents."""
        docs = operations.get_all_documents()
        return render_template("documents.html", active_page="documents", documents=docs)

    @app.route("/documents/<int:doc_id>")
    def document_detail(doc_id: int):
        """View document metadata and passage chunks."""
        doc = operations.get_document_by_id(doc_id)
        if not doc:
            abort(404, "Document not found")
        chunks = operations.get_chunks_for_document(doc_id)
        return render_template(
            "document_detail.html",
            active_page="documents",
            document=doc,
            chunks=chunks
        )

    @app.route("/documents/<int:doc_id>/delete", methods=["POST"])
    def delete_document_action(doc_id: int):
        """Delete document by ID and cascade delete chunks."""
        doc = operations.get_document_by_id(doc_id)
        if not doc:
            flash("Document not found.", "warning")
            return redirect(url_for("documents_page"))

        # Delete from disk if exists
        try:
            file_path = Config.UPLOAD_DIR / doc["filename"]
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

        operations.delete_document(doc_id)
        rebuild_all_indexes()
        flash(f"Document '{doc['title']}' deleted successfully.", "success")
        return redirect(url_for("documents_page"))

    @app.route("/documents/<int:doc_id>/summarize")
    def summarize_action(doc_id: int):
        """Generate executive summary for a specific document."""
        doc = operations.get_document_by_id(doc_id)
        if not doc:
            abort(404, "Document not found")
            
        summary_result = summarizer_service.summarize_document(
            text=doc["full_text"],
            title=doc["title"]
        )
        chunks = operations.get_chunks_for_document(doc_id)
        return render_template(
            "document_detail.html",
            active_page="documents",
            document=doc,
            chunks=chunks,
            summary=summary_result["summary"],
            summary_method=summary_result["method"]
        )

    @app.route("/attention")
    def attention_page():
        """Transformer Attention Mechanism laboratory."""
        default_sentence = "Artificial intelligence detects disease early in healthcare analytics."
        attn_data = attention_service.extract_attention(default_sentence, layer_idx=0, head_idx=0)
        return render_template(
            "attention.html",
            active_page="attention",
            **attn_data
        )

    @app.route("/attention/extract")
    def attention_action():
        """Extract real attention weights and render heatmap."""
        sentence = request.args.get("sentence", "").strip()
        layer_param = int(request.args.get("layer", 1)) - 1
        head_param_str = request.args.get("head", "1")
        
        head_idx = None if head_param_str == "avg" else int(head_param_str) - 1
        attn_data = attention_service.extract_attention(
            sentence=sentence,
            layer_idx=layer_param,
            head_idx=head_idx
        )
        return render_template(
            "attention.html",
            active_page="attention",
            **attn_data
        )

    @app.route("/word2vec")
    def word2vec_page():
        """Word2Vec laboratory for similarity and nearest neighbors."""
        action = request.args.get("action", "")
        arch = request.args.get("arch", "cbow")
        
        pair_similarity = None
        neighbors = None
        word1 = request.args.get("word1", "disease")
        word2 = request.args.get("word2", "healthcare")
        target_word = request.args.get("target_word", "learning")

        if action == "similarity" and word1 and word2:
            pair_similarity = word2vec_service.compute_similarity(word1, word2, architecture=arch)
            if pair_similarity is None:
                flash(f"Words '{word1}' or '{word2}' not found in vocabulary.", "info")
                
        elif action == "neighbors" and target_word:
            neighbors = word2vec_service.find_most_similar([target_word], architecture=arch, top_n=6)
            if not neighbors:
                flash(f"Word '{target_word}' not found in vocabulary.", "info")

        vocab = word2vec_service.get_vocabulary(architecture=arch)
        return render_template(
            "word2vec.html",
            active_page="word2vec",
            vocab_size=len(vocab),
            arch=arch,
            word1=word1,
            word2=word2,
            target_word=target_word,
            pair_similarity=pair_similarity,
            neighbors=neighbors
        )

    @app.route("/word2vec/train", methods=["POST"])
    def train_word2vec_action():
        """Retrain Word2Vec CBOW and Skip-gram on current corpus."""
        chunks = operations.get_all_chunks()
        texts = [c["chunk_text"] for c in chunks]
        res = word2vec_service.train(texts)
        flash(f"Word2Vec retrained successfully! Vocabulary: {res.get('vocabulary_size', 0)} terms.", "success")
        return redirect(url_for("word2vec_page"))

    @app.route("/clustering")
    def clustering_page():
        """K-Means clustering and PCA visualization."""
        k = int(request.args.get("k", 3))
        method = request.args.get("method", "tfidf")
        docs = operations.get_all_documents()
        
        cluster_result = clustering_service.cluster_documents(docs, num_clusters=k, method=method)
        return render_template(
            "clustering.html",
            active_page="clustering",
            k=k,
            method=method,
            cluster_result=cluster_result
        )

    @app.route("/visualizations")
    def visualizations_page():
        """Linguistic Visualizations: Word Cloud, Bigram Network Graph, and Frequency Bar Chart."""
        doc_id_param = int(request.args.get("doc_id", 0))
        bar_type = request.args.get("bar_type", "unigram")
        top_n = int(request.args.get("top_n", 15))

        all_docs = operations.get_all_documents()
        
        # Determine text scope to analyze
        if doc_id_param > 0:
            doc = operations.get_document_by_id(doc_id_param)
            if doc:
                text_to_analyze = doc["full_text"]
                current_doc_title = doc["title"]
            else:
                doc_id_param = 0
                text_to_analyze = " ".join([d["full_text"] for d in all_docs])
                current_doc_title = "All Documents"
        else:
            text_to_analyze = " ".join([d["full_text"] for d in all_docs])
            current_doc_title = "All Documents"

        cleaned_tokens = visualization_service.extract_cleaned_tokens(text_to_analyze)
        word_count = len(cleaned_tokens)

        # 1. Generate Word Cloud
        wordcloud_img, top_words = visualization_service.generate_word_cloud(text_to_analyze, max_words=80)
        
        # 2. Generate Bigram Network Graph
        network_img, top_bigrams = visualization_service.generate_bigram_network(text_to_analyze, top_n=top_n)
        
        # 3. Generate Frequency Bar Chart
        barchart_img, bar_data = visualization_service.generate_barchart(text_to_analyze, chart_type=bar_type, top_k=top_n)

        return render_template(
            "visualizations.html",
            active_page="visualizations",
            all_documents=all_docs,
            selected_doc_id=doc_id_param,
            current_doc_title=current_doc_title,
            bar_type=bar_type,
            top_n=top_n,
            word_count=word_count,
            wordcloud_img=wordcloud_img,
            top_words=top_words,
            network_img=network_img,
            top_bigrams=top_bigrams,
            barchart_img=barchart_img,
            bar_data=bar_data
        )

    @app.route("/analytics")
    def analytics_page():
        """Analytics dashboard and empirical benchmark logs."""
        stats = operations.get_database_stats()
        eval_metrics = operations.get_latest_evaluation_metrics()
        search_history = operations.get_search_history(limit=30)
        return render_template(
            "analytics.html",
            active_page="analytics",
            stats=stats,
            eval_metrics=eval_metrics,
            search_history=search_history
        )

    @app.route("/analytics/run-evaluation", methods=["POST"])
    def run_eval_action():
        """Execute automated benchmark evaluation across all 4 modes."""
        evaluator_service.run_full_benchmark(top_k=5)
        flash("Benchmark evaluation completed and metrics updated.", "success")
        return redirect(url_for("analytics_page"))

    @app.route("/settings")
    def settings_page():
        """Settings page."""
        return render_template("settings.html", active_page="settings", config=Config)

    @app.route("/settings/save", methods=["POST"])
    def save_settings_action():
        """Update runtime settings."""
        Config.DEFAULT_SEARCH_MODE = request.form.get("default_search_mode", Config.DEFAULT_SEARCH_MODE)
        Config.DEFAULT_TOP_K = int(request.form.get("default_top_k", Config.DEFAULT_TOP_K))
        Config.DEFAULT_HYBRID_ALPHA = float(request.form.get("default_alpha", Config.DEFAULT_HYBRID_ALPHA))
        Config.LLM_PROVIDER = request.form.get("llm_provider", Config.LLM_PROVIDER)
        Config.OLLAMA_BASE_URL = request.form.get("ollama_base_url", Config.OLLAMA_BASE_URL)
        Config.OLLAMA_MODEL = request.form.get("ollama_model", Config.OLLAMA_MODEL)
        Config.OPENAI_API_KEY = request.form.get("openai_api_key", Config.OPENAI_API_KEY)
        
        # Update LLM service instance
        llm_service.provider = Config.LLM_PROVIDER
        llm_service.ollama_base_url = Config.OLLAMA_BASE_URL.rstrip("/")
        llm_service.ollama_model = Config.OLLAMA_MODEL
        llm_service.openai_api_key = Config.OPENAI_API_KEY
        
        flash("System configuration updated successfully.", "success")
        return redirect(url_for("settings_page"))

    @app.route("/settings/rebuild", methods=["POST"])
    def rebuild_indexes_action():
        """Rebuild all in-memory indexes and re-cache embeddings."""
        rebuild_all_indexes()
        flash("All indexes (Inverted Index, TF-IDF, Semantic Embeddings, Word2Vec) rebuilt successfully.", "success")
        return redirect(url_for("settings_page"))

    @app.route("/settings/reset-sample", methods=["POST"])
    def reset_sample_data_action():
        """Regenerate sample data."""
        from scripts.create_sample_data import generate_sample_documents
        generate_sample_documents()
        rebuild_all_indexes()
        flash("Sample dataset regenerated and all indexes updated.", "success")
        return redirect(url_for("documents_page"))

    @app.route("/health")
    def health_check():
        """API Health check endpoint."""
        stats = operations.get_database_stats()
        return jsonify({
            "status": "healthy",
            "service": "AI-Powered Semantic Search Engine",
            "database_connected": True,
            "document_count": stats.get("total_documents", 0),
            "chunk_count": stats.get("total_chunks", 0),
            "embedding_model": Config.EMBEDDING_MODEL_NAME,
            "llm_provider": Config.LLM_PROVIDER
        })

    return app

if __name__ == "__main__":
    app = create_app()
    print("=" * 70)
    print("STARTING AI-POWERED SEMANTIC SEARCH ENGINE WEB APPLICATION")
    print("Access the dashboard in your web browser at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)
