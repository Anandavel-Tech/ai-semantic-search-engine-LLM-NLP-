# AI-Powered Semantic Search Engine Using LLMs and Attention Mechanisms

An end-to-end, production-ready Natural Language Processing (NLP) and Information Retrieval (IR) system built with Python, Flask, PyMuPDF, Scikit-Learn, Gensim, Hugging Face Transformers, Sentence Transformers, and SQLite.

Engineered for enterprise document intelligence, technical evaluation, and production-grade NLP engineering.

---

## Key Highlights

- **Dual-Engine Retrieval**: Lexical keyword matching (Scikit-Learn TF-IDF & custom Inverted Index with positional postings) alongside dense contextual semantic retrieval (Sentence Transformers `all-MiniLM-L6-v2`).
- **Configurable Hybrid Ranking**: Linear score combination with Min-Max calibration and Reciprocal Rank Fusion (RRF).
- **Retrieval-Augmented Generation (RAG)**: Context-grounded Question Answering with strict source citations, prompt-injection defense guardrails, and an intelligent extractive fallback engine that functions 100% offline.
- **Real Transformer Attention Visualization**: Extracts real multi-head self-attention matrices ($A = \text{softmax}(QK^\top / \sqrt{d_k})$) across user-selected layers and heads, generating dynamic heatmaps.
- **Word2Vec Laboratory**: Continuous Bag-of-Words (CBOW) and Skip-gram neural embeddings with vocabulary exploration, cosine similarity, and vector analogies.
- **Unsupervised Document Clustering**: K-Means clustering on TF-IDF or dense embeddings, 2D PCA projection canvas visualization, and Silhouette Coefficient evaluation.
- **Linguistic Visualizations & Graphs**: Interactive **Word Cloud** generation, **Bigram Co-occurrence Network Graphs** (NetworkX spring layout), and **Term Frequency Bar Charts** (Unigrams/Bigrams) across the entire corpus or per-document.
- **Empirical Evaluation Pipeline**: Automated benchmark comparison measuring Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Latency across 8 benchmark queries.
- **PyMuPDF Document Ingestion**: Ingests multi-page PDFs and TXT files, enforces SHA-256 deduplication, and generates overlapping sliding-window passages (200 words / 40-word overlap).

---

## Project Structure

```text
NLP/
├── app.py                     # Main Flask web application & REST routes
├── config.py                  # Central configuration & path management
├── requirements.txt           # Compatible dependency specifications
├── .env.example               # Environment variables template
├── .gitignore                 # Version control exclusion rules
├── database/
│   ├── __init__.py
│   ├── db.py                  # SQLite connection & schema initialization
│   ├── models.py / schema.sql # Relational schema with cascading foreign keys
│   └── operations.py          # Thread-safe, parameterized CRUD operations
├── services/
│   ├── __init__.py
│   ├── preprocessor.py        # Normalization, tokenization, stemming, lemmatization
│   ├── document_processor.py  # PyMuPDF PDF/TXT parser, chunking, deduplication
│   ├── inverted_index.py      # Positional inverted index & postings search
│   ├── tfidf_search.py        # Scikit-learn TF-IDF & cosine similarity ranking
│   ├── embeddings_service.py  # SentenceTransformers (all-MiniLM-L6-v2) & caching
│   ├── hybrid_search.py       # Linear combination (alpha) & RRF fusion
│   ├── word2vec_service.py    # Gensim CBOW and Skip-gram exploration
│   ├── attention_service.py   # Transformer attention extraction & heatmap generation
│   ├── clustering_service.py  # K-Means clustering, 2D PCA & silhouette score
│   ├── llm_service.py         # RAG pipeline, citations & extractive fallback QA
│   ├── summarization_service.py # Single & multi-document synthesis
│   └── evaluation_service.py  # IR metrics (Precision@K, Recall@K, MRR, Latency)
├── templates/                 # Bootstrap 5 responsive UI templates
│   ├── base.html              # Layout, navbar, and flash alerts
│   ├── index.html             # Search portal with score breakdowns
│   ├── upload.html            # Drag-and-drop PDF/TXT document upload
│   ├── documents.html         # Document manager & chunk overview
│   ├── document_detail.html   # Metadata, chunk inspector & summary
│   ├── qa.html                # RAG Question-Answering interface
│   ├── attention.html         # Transformer attention heatmap visualizer
│   ├── word2vec.html          # Word2Vec laboratory (CBOW vs Skip-Gram)
│   ├── clustering.html        # K-Means & PCA 2D scatter visualizer
│   ├── analytics.html         # Evaluation benchmarks & query latency history
│   └── settings.html          # Runtime parameters & index rebuilding
├── static/
│   ├── css/custom.css         # Modern styling & responsive theme
│   └── js/main.js             # Slider reactivity & HTML5 canvas scatter plot
├── scripts/
│   ├── create_sample_data.py  # Generates 24 academic demonstration documents
│   ├── run_evaluation.py      # Evaluates benchmark IR metrics across modes
│   └── download_nltk.py       # Pre-caches NLTK corpora
├── tests/                     # Comprehensive automated pytest suite (44 tests)
│   ├── test_preprocessor.py
│   ├── test_document_processor.py
│   ├── test_indexing_and_search.py
│   ├── test_attention_and_clustering.py
│   ├── test_llm_and_summarization.py
│   ├── test_database_and_operations.py
│   └── test_flask_routes.py
└── docs/                      # Academic documentation
    ├── ARCHITECTURE.md        # Architecture diagrams & component flows
    ├── ALGORITHMS.md          # Full mathematical & algorithmic formulations
    └── USER_GUIDE.md          # Step-by-step evaluator walkthrough
```

---

## Installation & Windows Execution Guide

### Prerequisites
- Windows 10 or 11 (64-bit)
- Python 3.11 or Python 3.12 (64-bit)
- Git (optional)

### Step 1: Clone or Navigate to Directory
```powershell
cd c:\Users\balut\OneDrive\Desktop\NLP
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Step 3: Upgrade pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch (CPU-Optimized)**:  
> If installing on a machine without a dedicated GPU, install the lightweight CPU build:  
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Step 4: Pre-cache NLTK Resources
```powershell
python scripts/download_nltk.py
```

### Step 5: Initialize Database and Generate 24 Sample Academic Documents
```powershell
python scripts/create_sample_data.py
```
This generates 24 demonstration documents in both PDF and TXT formats covering Healthcare AI, Transformers, NLP Preprocessing, Cybersecurity, Climate Science, Robotics, and Computer Vision.

### Step 6: Run Empirical Search Benchmark
```powershell
python scripts/run_evaluation.py
```
This tests all 4 retrieval modes against 8 benchmark queries and outputs the comparative Precision@5, Recall@5, MRR, and Latency table.

### Step 7: Launch the Web Application
```powershell
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## Running the Automated Test Suite

Execute the complete pytest test suite:
```powershell
pytest -v
```
All 44 test cases across unit, integration, service, and web route layers will run and pass:
```text
======================= 44 passed =======================
```

---

## Optional: Configuring Local LLM (Ollama)

The application includes an **Extractive Fallback QA engine** that requires **no external API or internet access**.

If you wish to connect a local LLM via [Ollama](https://ollama.ai/):
1. Install Ollama and pull your desired model:
   ```powershell
   ollama pull llama3.2
   ```
2. Navigate to **Settings** (`/settings`) in the web interface.
3. Change **Active Provider** to `Local Ollama Server` and set the model name to `llama3.2`.
4. Click **Save Configuration**.

---

## Troubleshooting

| Issue | Resolution |
| :--- | :--- |
| **Port 5000 already in use** | In `app.py`, change `app.run(port=5000)` to `port=5001`. |
| **PyMuPDF deprecation warning** | The project explicitly uses the modern `import pymupdf` standard API. |
| **Hugging Face SSL / Corporate Firewall** | The application automatically falls back to an exact mathematical Scaled Dot-Product Attention engine and LSA semantic projector without crashing. |
| **Duplicate file warning on upload** | The system enforces SHA-256 deduplication to prevent redundant chunks. Uploading a modified file will proceed normally. |
